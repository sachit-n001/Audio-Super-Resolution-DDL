import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Load only the model definitions, without running training.
source = Path("train_audio_benchmark.py").read_text()
prefix = source.split("if not torch.cuda.is_available():")[0]
exec(compile(prefix, "benchmark_definitions", "exec"), globals())

device = torch.device("cuda")
checkpoint = torch.load(
    "runs/62043192/checkpoint.pt",
    map_location="cpu",
    weights_only=False,
)
model = AudioDenoiser().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

validation = torch.tensor(
    np.load("data/multiclip_pilot/validation.npy", allow_pickle=False),
    dtype=torch.float32,
    device=device,
)
assert validation.shape[1] == LENGTH
assert torch.isfinite(validation).all()

frequencies = torch.fft.rfftfreq(LENGTH, d=1 / RATE, device=device)
mask = frequencies >= RATE / 4

def highpass(x):
    return torch.fft.irfft(torch.fft.rfft(x) * mask, n=LENGTH)

low = validation - highpass(validation)
residual_scale = checkpoint["metrics"]["residual_scale"]
condition_scale = checkpoint["metrics"]["condition_scale"]
condition = low / condition_scale

beta = checkpoint["beta"].to(device)
alpha_bar = (1 - beta).cumprod(0)
baseline = (low - validation).square().mean().item()

out = Path("runs") / ("sampling_" + os.environ["SLURM_JOB_ID"])
out.mkdir(parents=True, exist_ok=False)

@torch.no_grad()
def generate(count):
    # DDIM with eta=0: use the trained schedule with fewer selected steps.
    indices = torch.linspace(
        len(beta) - 1, 0, count, device=device
    ).round().long()
    assert len(indices.unique()) == count

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    x = torch.randn_like(low)

    for position, index in enumerate(indices):
        i = int(index.item())
        t = torch.full(
            (len(validation),), i, dtype=torch.long, device=device
        )
        clean = model(x, condition, t)
        current = alpha_bar[i]
        noise = (x - current.sqrt() * clean) / (1 - current).sqrt()

        previous = (
            alpha_bar[indices[position + 1]]
            if position + 1 < count
            else torch.ones((), device=device)
        )
        x = previous.sqrt() * clean + (1 - previous).sqrt() * noise

    return low + highpass(x * residual_scale)

rows = []
for count in (10, 25, 50, 100):
    generate(count)  # Untimed warmup.
    durations = []

    for repeat in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        reconstructed = generate(count)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)

    assert torch.isfinite(reconstructed).all()
    mse = (reconstructed - validation).square().mean().item()
    seconds = float(np.median(durations))

    row = {
        "sampling_steps": count,
        "sampler": "DDIM_eta0",
        "validation_clips": len(validation),
        "median_batch_seconds": seconds,
        "milliseconds_per_clip": 1000 * seconds / len(validation),
        "baseline_mse": baseline,
        "reconstruction_mse": mse,
        "improvement_percent": 100 * (1 - mse / baseline),
    }
    rows.append(row)
    np.save(
        out / f"reconstruction_{count}steps.npy",
        reconstructed.cpu().numpy(),
    )
    print(json.dumps(row), flush=True)

with (out / "sampling_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(out / "metadata.json").write_text(json.dumps({
    "checkpoint": "runs/62043192/checkpoint.pt",
    "gpu": torch.cuda.get_device_name(),
    "torch_version": str(torch.__version__),
    "sampling_seed": 1234,
    "timing_repeats": 3,
    "sampler": "DDIM_eta0",
    "note": "All step counts use DDIM; earlier training runs used DDPM.",
}, indent=2))

print(f"EVALUATION COMPLETE: {out}", flush=True)
