import ast
import csv
import json
import math
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Reuse the same nanoGPT backbone and audio model as the successful pilot.
repo = Path(os.environ["SCRATCH"]) / "audio-super-resolution-nanogpt"
sys.path.insert(0, str(repo))
from model import Block, GPTConfig

RATE, PATCH, PATCHES, WIDTH, STEPS = 44100, 128, 64, 256, 100
LENGTH = PATCH * PATCHES
BATCH = 128
UPDATES = 10000
WARMUP = 50

model_source = Path("train_audio_scaled.py").read_text()
assert model_source.count("n_layer=3,") == 1
model_source = model_source.replace("n_layer=3,", "n_layer=6,", 1)
tree = ast.parse(model_source)
model_class = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "AudioDenoiser"
)
exec(compile(ast.Module(body=[model_class], type_ignores=[]),
             "train_audio_scaled.py", "exec"))

if not torch.cuda.is_available():
    raise RuntimeError("Run this script through the GPU batch job.")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import builtins
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
assert BATCH % world == 0
local_batch = BATCH // world

# Slurm may expose one assigned GPU or all GPUs on the node.
visible = torch.cuda.device_count()
assert visible == world, (
    f"Expected {world} visible GPUs, found {visible}; "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
)
gpu_index = local_rank
torch.cuda.set_device(gpu_index)
device = torch.device("cuda", gpu_index)

dist.init_process_group(
    backend="nccl", rank=rank, world_size=world
)

# Keep the shared log readable.
def print(*args, **kwargs):
    if rank == 0:
        builtins.print(*args, **kwargs)

out = Path("runs") / (
    "scaling50_" + os.environ["SLURM_JOB_ID"]
)
out.mkdir(parents=True, exist_ok=True)

data_dir = Path("data/multiclip_50songs")

def load_split(name):
    array = np.load(data_dir / f"{name}.npy", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != LENGTH:
        raise ValueError(f"Unexpected {name} shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite samples in {name}")
    return torch.tensor(array, dtype=torch.float32, device=device)

train = load_split("train")
validation = load_split("validation")

manifest = json.loads((data_dir / "manifest.json").read_text())
train_songs = {item["song"] for item in manifest["train"]}
val_songs = {item["song"] for item in manifest["validation"]}
assert train_songs.isdisjoint(val_songs), "Song leakage!"

freqs = torch.fft.rfftfreq(LENGTH, d=1 / RATE, device=device)
high_mask = freqs >= RATE / 4

def highpass(x):
    return torch.fft.irfft(
        torch.fft.rfft(x) * high_mask, n=LENGTH
    )

train_high = highpass(train)
train_low = train - train_high
val_high = highpass(validation)
val_low = validation - val_high

# Both scales come exclusively from training data.
residual_scale = train_high.square().mean().sqrt().clamp_min(1e-6)
condition_scale = train_low.square().mean().sqrt().clamp_min(1e-6)
clean_train = train_high / residual_scale
condition_train = train_low / condition_scale

# Cosine diffusion schedule: terminal training noise is close to pure noise.
grid = torch.linspace(0, STEPS, STEPS + 1, device=device) / STEPS
curve = torch.cos((grid + 0.008) / 1.008 * math.pi / 2).square()
curve = curve / curve[0]
beta = (1 - curve[1:] / curve[:-1]).clamp(1e-5, 0.999)
alpha = 1 - beta
alpha_bar = alpha.cumprod(0)

model = AudioDenoiser().to(device)
model = DDP(model, device_ids=[gpu_index])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"Train clips: {len(train)}; validation clips: {len(validation)}")
print(f"Global batch: {BATCH}; updates: {UPDATES}; warmup: {WARMUP}")
print(f"Residual scale: {residual_scale.item():.9e}", flush=True)

model.train()
loss_rows = []
torch.cuda.synchronize()
dist.barrier()
training_start = time.perf_counter()

for step in range(UPDATES):
    if step == WARMUP:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        measured_start = time.perf_counter()

    # Sample with replacement from the small pilot training dataset.
    indices = torch.randint(len(train), (BATCH,), device=device)
    clean = clean_train[indices]
    condition = condition_train[indices]
    t = torch.randint(STEPS, (BATCH,), device=device)
    a = alpha_bar[t, None]
    noisy = a.sqrt() * clean + (1 - a).sqrt() * torch.randn_like(clean)

    # Each rank generates the same global batch, then trains its slice.
    # This preserves the pilot's global sampling and noise sequence.
    part = slice(rank * local_batch, (rank + 1) * local_batch)
    clean = clean[part]
    condition = condition[part]
    noisy = noisy[part]
    t = t[part]
    prediction = model(noisy, condition, t)
    loss = (prediction - clean).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step == 0 or (step + 1) % 100 == 0:
        reported_loss = loss.detach().clone()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
        value = (reported_loss / world).item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite training loss")
        loss_rows.append((step + 1, value))
        print(f"step={step + 1} loss={value:.8e}", flush=True)

torch.cuda.synchronize()
training_end = time.perf_counter()
measured_seconds = training_end - measured_start
peak_gib = torch.cuda.max_memory_allocated() / 2**30

# Report the slowest worker and maximum allocated memory across GPUs.
summary = torch.tensor(
    [measured_seconds, peak_gib, training_end - training_start],
    dtype=torch.float64, device=device,
)
dist.all_reduce(summary, op=dist.ReduceOp.MAX)
measured_seconds, peak_gib, total_training_seconds = summary.tolist()

# All workers have synchronized weights. Only rank zero evaluates and saves.
model = model.module
dist.barrier()
dist.destroy_process_group()
if rank != 0:
    sys.exit(0)

# Evaluate using degraded validation inputs and a fixed sampling seed.
# Validation targets are used only to calculate metrics.
model.eval()
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
with torch.no_grad():
    condition = val_low / condition_scale
    x = torch.randn_like(val_low)
    for i in reversed(range(STEPS)):
        t = torch.full((len(validation),), i, device=device, dtype=torch.long)
        clean = model(x, condition, t)
        previous = alpha_bar[i - 1] if i > 0 else torch.ones((), device=device)
        denominator = 1 - alpha_bar[i]
        mean = (
            previous.sqrt() * beta[i] / denominator * clean
            + alpha[i].sqrt() * (1 - previous) / denominator * x
        )
        if i > 0:
            variance = beta[i] * (1 - previous) / denominator
            x = mean + variance.sqrt() * torch.randn_like(x)
        else:
            x = mean

    reconstructed = val_low + highpass(x * residual_scale)
    if not torch.isfinite(reconstructed).all():
        raise RuntimeError("Nonfinite validation output")

    baseline = (val_low - validation).square().mean().item()
    reconstruction = (reconstructed - validation).square().mean().item()
    low_error = (
        (reconstructed - validation) - highpass(reconstructed - validation)
    ).square().mean().item()

revision = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
metrics = {
    "gpu": torch.cuda.get_device_name(),
    "torch_version": str(torch.__version__),
    "nanogpt_revision": revision,
    "world_size": world,
    "local_batch": local_batch,
    "repeat": int(os.environ["SLURM_ARRAY_TASK_ID"]),
    "layers": 6,
    "width": WIDTH,
    "heads": 4,
    "training_seed": 42,
    "global_batch": BATCH,
    "updates": UPDATES,
    "warmup_updates": WARMUP,
    "measured_training_seconds": measured_seconds,
    "mean_step_ms": 1000 * measured_seconds / (UPDATES - WARMUP),
    "clips_per_second": BATCH * (UPDATES - WARMUP) / measured_seconds,
    "training_seconds_including_warmup": total_training_seconds,
    "peak_allocated_gib_after_warmup": peak_gib,
    "train_clips": len(train),
    "validation_clips": len(validation),
    "validation_baseline_mse": baseline,
    "validation_reconstruction_mse": reconstruction,
    "validation_low_band_mse": low_error,
    "validation_improvement_percent": 100 * (1 - reconstruction / baseline)
        if baseline > 0 else None,
    "residual_scale": residual_scale.item(),
    "condition_scale": condition_scale.item(),
}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
with (out / "loss.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", "training_loss"])
    writer.writerows(loss_rows)

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "metrics": metrics,
    "beta": beta.cpu(),
    "config": {"rate": RATE, "patch": PATCH, "patches": PATCHES,
               "width": WIDTH, "diffusion_steps": STEPS,
               "layers": 6, "heads": 4},
}, out / "checkpoint.pt")

for name, signal in [
    ("validation_target", validation[0]),
    ("validation_degraded", val_low[0]),
    ("validation_reconstructed", reconstructed[0]),
]:
    pcm = (signal.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
    with wave.open(str(out / f"{name}.wav"), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.astype("<i2").tobytes())

print(json.dumps(metrics, indent=2), flush=True)
print(f"RUN COMPLETE: {out}", flush=True)
print("Quality must be assessed from validation metrics; completion is not a quality pass.")
