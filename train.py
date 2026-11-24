import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.audio_transformer import GPT, GPTConfig
from scripts.audio_dataset import AudioDataset


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=0)

    args = parser.parse_args()

    # Distributed training setup
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)

    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    master = rank == 0

    torch.manual_seed(42)

    # Keep global batch size fixed across GPU configurations
    if args.batch_size % world_size != 0:
        raise ValueError(
            "Global batch size must be divisible by GPU count"
        )

    local_batch_size = args.batch_size // world_size

    # Load datasets
    train_dataset = AudioDataset(
        data_dir=args.data_dir,
        split="train"
    )

    val_dataset = AudioDataset(
        data_dir=args.data_dir,
        split="val"
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if master else None

    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is too small")

    # Initialize transformer
    config = GPTConfig()
    model = GPT(config).to(device)

    if master:
        print(f"GPUs: {world_size}", flush=True)
        print(f"Global batch size: {args.batch_size}", flush=True)
        print(f"Local batch size: {local_batch_size}", flush=True)
        print(f"Parameters: {model.get_num_params():,}", flush=True)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    if master:
        Path(args.out_dir).mkdir(
            parents=True, exist_ok=True
        )

    best_val_loss = float("inf")
    global_step = 0
    total_samples = 0
    total_train_time = 0.0

    # Training loop
    for epoch in range(args.epochs):

        if distributed:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        model.train()

        epoch_loss = torch.zeros(1, device=device)
        epoch_samples = 0
        epoch_steps = 0

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for low, high in train_loader:

            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            reconstructed, loss = model(low, high)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            epoch_loss += loss.detach() * low.size(0)
            epoch_samples += low.size(0)
            epoch_steps += 1
            global_step += 1

            if master and global_step % 50 == 0:
                print(
                    f"Step {global_step} | "
                    f"Local loss: {loss.item():.6f}",
                    flush=True
                )

            if args.max_steps > 0:
                if global_step >= args.max_steps:
                    break

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        # Aggregate results across GPUs
        stats = torch.tensor(
            [epoch_samples],
            dtype=torch.float64,
            device=device
        )

        elapsed_tensor = torch.tensor(
            [elapsed],
            dtype=torch.float64,
            device=device
        )

        if distributed:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(
                elapsed_tensor,
                op=dist.ReduceOp.MAX
            )

        samples = int(stats.item())
        elapsed = elapsed_tensor.item()

        train_loss = epoch_loss.item() / samples
        throughput = samples / elapsed

        total_samples += samples
        total_train_time += elapsed

        # Validation on the main GPU
        if master:

            raw_model.eval()

            val_loss_sum = 0.0
            val_samples = 0

            with torch.no_grad():

                for low, high in val_loader:

                    low = low.to(device)
                    high = high.to(device)

                    _, loss = raw_model(low, high)

                    val_loss_sum += loss.item() * low.size(0)
                    val_samples += low.size(0)

            val_loss = val_loss_sum / val_samples

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train loss: {train_loss:.6f} | "
                f"Val loss: {val_loss:.6f} | "
                f"Time: {elapsed:.2f}s | "
                f"Throughput: {throughput:.2f} samples/s",
                flush=True
            )

            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(config),
                "train_loss": train_loss,
                "val_loss": val_loss
            }

            torch.save(
                checkpoint,
                Path(args.out_dir) / "latest.pt"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    checkpoint,
                    Path(args.out_dir) / "best.pt"
                )

        if distributed:
            dist.barrier()

        if args.max_steps > 0:
            if global_step >= args.max_steps:
                break

    # Final performance summary
    if master:

        print("\nTraining complete", flush=True)
        print(f"GPU count: {world_size}", flush=True)
        print(f"Total steps: {global_step}", flush=True)
        print(f"Total samples: {total_samples}", flush=True)

        print(
            f"Training time: {total_train_time:.2f}s",
            flush=True
        )

        print(
            f"Average throughput: "
            f"{total_samples / total_train_time:.2f} samples/s",
            flush=True
        )

        print(
            f"Best validation loss: {best_val_loss:.6f}",
            flush=True
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()