import ast
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

mode = sys.argv[1]
if mode not in {"robustness", "spectral"}:
    raise ValueError("Choose robustness or spectral")
if not torch.cuda.is_available():
    raise RuntimeError("Run through the GPU batch job")

selection = json.loads(Path("analysis/selected_checkpoint.json").read_text())
checkpoint_path = Path(selection["checkpoint"])
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=False
)
config = checkpoint["config"]
RATE = int(config["rate"])
PATCH = int(config["patch"])
PATCHES = int(config["patches"])
WIDTH = int(config["width"])
STEPS = int(config["diffusion_steps"])
LENGTH = PATCH * PATCHES
LAYERS = int(config["layers"])
HEADS = int(config["heads"])

repo = Path(os.environ["SCRATCH"]) / "audio-super-resolution-nanogpt"
sys.path.insert(0, str(repo))
from model import Block, GPTConfig

# Load only the model class, without executing the training script.
source = Path("train_audio_scaled.py").read_text()
tree = ast.parse(source)
model_class = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "AudioDenoiser"
)

class ConfigureModel(ast.NodeTransformer):
    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "n_layer":
            node.value = ast.Constant(LAYERS)
        elif node.arg == "n_head":
            node.value = ast.Constant(HEADS)
        return node

model_class = ConfigureModel().visit(model_class)
module = ast.Module(body=[model_class], type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, "train_audio_scaled.py", "exec"))

device = torch.device("cuda")
model = AudioDenoiser().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

data_dir = Path("data/final_test")
validation = torch.from_numpy(
    np.load(data_dir / "test.npy", allow_pickle=False)
).float().to(device)

if validation.ndim != 2 or validation.shape[1] != LENGTH:
    raise RuntimeError("Unexpected validation shape")
if not torch.isfinite(validation).all():
    raise RuntimeError("Nonfinite validation data")

manifest = json.loads((data_dir / "manifest.json").read_text())
records = manifest["test"]
assert len(records) == len(validation)

frequencies = torch.fft.rfftfreq(LENGTH, d=1 / RATE, device=device)
mask = frequencies >= RATE / 4

def highpass(signal):
    return torch.fft.irfft(
        torch.fft.rfft(signal) * mask, n=LENGTH
    )

low = validation - highpass(validation)
residual_scale = float(checkpoint["metrics"]["residual_scale"])
condition_scale = float(checkpoint["metrics"]["condition_scale"])
beta = checkpoint["beta"].to(device)
alpha = 1 - beta
alpha_bar = alpha.cumprod(0)

@torch.no_grad()
def reconstruct(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    condition = low / condition_scale
    x = torch.randn_like(low)

    for i in reversed(range(STEPS)):
        t = torch.full(
            (len(validation),), i, device=device, dtype=torch.long
        )
        clean = model(x, condition, t)
        previous = (
            alpha_bar[i - 1] if i > 0
            else torch.ones((), device=device)
        )
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

    result = low + highpass(x * residual_scale)
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite reconstruction")
    return result.cpu().numpy()

output = Path("analysis") / f"final_test_{os.environ['SLURM_JOB_ID']}"
output.mkdir(parents=True, exist_ok=False)
target = validation.cpu().numpy().astype(np.float64)
degraded = low.cpu().numpy().astype(np.float64)
baseline = float(np.mean((degraded - target) ** 2))

def save_csv(name, rows):
    with (output / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if mode == "robustness":
    rows = []
    for seed in range(1234, 1244):
        result = reconstruct(seed).astype(np.float64)
        mse = float(np.mean((result - target) ** 2))
        rows.append({
            "sampling_seed": seed,
            "validation_clips": len(target),
            "baseline_mse": baseline,
            "reconstruction_mse": mse,
            "improvement_percent": 100 * (1 - mse / baseline),
        })
        print(rows[-1], flush=True)

    save_csv("sampling_seed_metrics.csv", rows)
    values = np.array([row["reconstruction_mse"] for row in rows])
    summary = {
        "sampling_seeds": len(rows),
        "mean_reconstruction_mse": float(values.mean()),
        "sample_std_reconstruction_mse": float(values.std(ddof=1)),
        "min_reconstruction_mse": float(values.min()),
        "max_reconstruction_mse": float(values.max()),
    }
else:
    result = reconstruct(1234).astype(np.float64)
    np.save(output / "reconstructed_test.npy", result)
    import wave
    for clip_index in range(min(3, len(target))):
        for label, samples in [
            ("target", target[clip_index]),
            ("degraded", degraded[clip_index]),
            ("reconstructed", result[clip_index]),
        ]:
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
            with wave.open(
                str(output / f"clip{clip_index}_{label}.wav"), "wb"
            ) as handle:
                handle.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                handle.writeframes(pcm.tobytes())


    def magnitude(signal):
        frames = np.lib.stride_tricks.sliding_window_view(
            signal, 1024
        )[::256]
        return np.abs(
            np.fft.rfft(frames * np.hanning(1024), axis=-1)
        )

    rows = []
    for index, reference in enumerate(target):
        reference_mag = magnitude(reference)
        for label, signal in [
            ("degraded", degraded[index]),
            ("reconstructed", result[index]),
        ]:
            signal_mag = magnitude(signal)
            db_difference = 20 * np.log10(
                np.maximum(signal_mag, 1e-7)
            ) - 20 * np.log10(np.maximum(reference_mag, 1e-7))
            rows.append({
                "clip": index,
                "song": records[index]["song"],
                "offset_seconds": records[index]["offset_seconds"],
                "signal": label,
                "mse": float(np.mean((signal - reference) ** 2)),
                "log_spectral_distance_db": float(
                    np.sqrt(np.mean(db_difference ** 2, axis=-1)).mean()
                ),
                "spectral_convergence": float(
                    np.linalg.norm(signal_mag - reference_mag)
                    / max(np.linalg.norm(reference_mag), 1e-12)
                ),
            })
    save_csv("test_clip_metrics.csv", rows)
    summary = {}
    for label in ["degraded", "reconstructed"]:
        selected = [row for row in rows if row["signal"] == label]
        summary[label] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in [
                "mse", "log_spectral_distance_db",
                "spectral_convergence",
            ]
        }

metadata = {
    "checkpoint": str(checkpoint_path),
    "mode": mode,
    "sampler": "DDPM",
    "sampling_steps": STEPS,
    "gpu": torch.cuda.get_device_name(),
    "torch_version": str(torch.__version__),
    "test_clips": len(target),
    "test_songs": len({record["song"] for record in records}),
    "spectral_settings": {
        "window": "numpy_hanning",
        "fft_size": 1024,
        "hop": 256,
        "magnitude_floor": 1e-7,
    },
    "summary": summary,
}
(output / "summary.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2), flush=True)
print(f"Saved results to {output}", flush=True)
