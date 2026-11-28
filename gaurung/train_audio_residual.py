import math
import os
import wave
from pathlib import Path

import torch
from torch import nn

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a GPU. Run this through Slurm.")
device = torch.device("cuda")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")

RATE = 44100
PATCH = 128
PATCHES = 64
LENGTH = PATCH * PATCHES
WIDTH = 128
STEPS = 100


class AudioDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        # Each input patch contains noisy residual audio and degraded audio.
        self.input = nn.Linear(2 * PATCH, WIDTH)
        self.position = nn.Parameter(torch.zeros(1, PATCHES, WIDTH))
        self.timestep = nn.Embedding(STEPS, WIDTH)
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=WIDTH, nhead=4, dim_feedforward=4 * WIDTH,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            )
            for _ in range(3)
        ])
        self.output = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, PATCH))

    def forward(self, noisy, degraded, t):
        noisy = noisy.reshape(-1, PATCHES, PATCH)
        degraded = degraded.reshape(-1, PATCHES, PATCH)
        x = self.input(torch.cat((noisy, degraded), dim=-1))
        x = x + self.position + self.timestep(t)[:, None, :]
        return self.output(self.blocks(x)).flatten(1)


# Make a toy pair. The degraded input is missing the 14 kHz tone.
seconds = torch.arange(LENGTH, device=device) / RATE
low = (0.4 * torch.sin(2 * math.pi * 440 * seconds))[None, :]
target = low + (0.2 * torch.sin(2 * math.pi * 14000 * seconds))[None, :]
residual = target - low

model = AudioDenoiser().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
beta = torch.linspace(0.0001, 0.06, STEPS, device=device)
alpha = 1 - beta
alpha_bar = alpha.cumprod(dim=0)

# Train the transformer to predict noise at randomly chosen timesteps.
for step in range(1, 501):
    t = torch.randint(STEPS, (8,), device=device)
    noise = torch.randn(8, LENGTH, device=device)
    a = alpha_bar[t, None]
    noisy = a.sqrt() * residual + (1 - a).sqrt() * noise
    prediction = model(noisy, low.expand_as(noisy), t)
    loss = (prediction - residual.expand_as(prediction)).square().mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite loss at step {step}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 1 or step % 100 == 0:
        print(f"step={step} residual_prediction_loss={loss.item():.6f}")

# Denoise from scratch, using only the degraded audio as the condition.
model.eval()
with torch.no_grad():
    x = torch.randn_like(low)
    for i in reversed(range(STEPS)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        clean = model(x, low, t)
        prediction = (x - alpha_bar[i].sqrt() * clean) / (1 - alpha_bar[i]).sqrt()
        x = (x - beta[i] * prediction / (1 - alpha_bar[i]).sqrt()) / alpha[i].sqrt()
        if i > 0:
            variance = beta[i] * (1 - alpha_bar[i - 1]) / (1 - alpha_bar[i])
            x = x + variance.sqrt() * torch.randn_like(x)
    reconstructed = low + x

if not torch.isfinite(reconstructed).all().item():
    raise RuntimeError("Reconstruction contains non-finite samples")

# Keep each job's outputs separate so reruns preserve earlier results.
out = Path("runs") / os.environ.get("SLURM_JOB_ID", "local")
out.mkdir(parents=True, exist_ok=True)
for name, signal in [
    ("degraded", low), ("target", target), ("reconstructed", reconstructed)
]:
    pcm = (signal.flatten().clamp(-1, 1) * 32767).to(torch.int16)
    with wave.open(str(out / f"{name}.wav"), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.cpu().numpy().astype("<i2").tobytes())

torch.save(model.state_dict(), out / "denoiser.pt")
print(f"Baseline MSE:       {(low - target).square().mean().item():.6f}")
print(f"Reconstruction MSE: {(reconstructed - target).square().mean().item():.6f}")
print(f"PASS: training and sampling completed; outputs in {out}/")
