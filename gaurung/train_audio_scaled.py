import math
import os
import wave
from pathlib import Path

import torch
from torch import nn

import importlib.util

# nanoGPT upstream revision: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
nanogpt_file = Path(os.environ["SCRATCH"]) / "audio-super-resolution-nanogpt" / "model.py"
spec = importlib.util.spec_from_file_location("audio_nanogpt_backbone", nanogpt_file)
nanogpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nanogpt)
Block = nanogpt.Block
GPTConfig = nanogpt.GPTConfig

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
        config = GPTConfig(
            block_size=PATCHES,
            n_layer=3,
            n_head=4,
            n_embd=WIDTH,
            dropout=0.0,
            bias=True,
        )
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layer)))
        self.output = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, PATCH))

    def forward(self, noisy, degraded, t):
        noisy = noisy.reshape(-1, PATCHES, PATCH)
        degraded = degraded.reshape(-1, PATCHES, PATCH)
        x = self.input(torch.cat((noisy, degraded), dim=-1))
        x = x + self.position + self.timestep(t)[:, None, :]
        return self.output(self.blocks(x)).flatten(1)


# Read one short clip from a real 44.1 kHz PCM WAV.
import numpy as np

wav_path = os.environ["AUDIO_WAV"]
with wave.open(wav_path, "rb") as wav:
    if wav.getframerate() != RATE or wav.getsampwidth() not in (2, 3):
        raise ValueError("Use a 44.1 kHz, 16-bit or 24-bit PCM WAV.")
    channels = wav.getnchannels()
    width = wav.getsampwidth()
    raw = wav.readframes(LENGTH)
if len(raw) != LENGTH * channels * width:
    raise ValueError(f"Input WAV must contain at least {LENGTH} frames.")

if width == 2:
    pcm = np.frombuffer(raw, dtype="<i2")
else:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    pcm = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    pcm = (pcm ^ 0x800000) - 0x800000

# Use the left channel for this mono pilot.
samples = pcm.reshape(-1, channels)[:, 0].astype(np.float32)
target = torch.tensor(samples / (2 ** (8 * width - 1)), device=device)[None, :]
# Remove frequencies above the 22.05 kHz input's Nyquist limit.
spectrum = torch.fft.rfft(target)
freqs = torch.fft.rfftfreq(LENGTH, d=1 / RATE, device=device)
low = torch.fft.irfft(spectrum * (freqs < RATE / 4), n=LENGTH)
print(f"Pilot audio: {wav_path}; {LENGTH / RATE:.3f} seconds")
# Scale estimated only from this pilot's training crop.
residual = target - low
residual_scale = residual.square().mean().sqrt().clamp_min(1e-6)
residual = residual / residual_scale
print(f"Training residual scale: {residual_scale.item():.8e}")

model = AudioDenoiser().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
beta = torch.linspace(0.0001, 0.06, STEPS, device=device)
alpha = 1 - beta
alpha_bar = alpha.cumprod(dim=0)

# Train the model to predict the clean residual at random diffusion timesteps.
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
        print(f"step={step} residual_prediction_loss={loss.item():.9e}")

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
    # Restore amplitude and retain only the missing frequency band.
    generated_residual = x * residual_scale
    generated_residual = torch.fft.irfft(
        torch.fft.rfft(generated_residual) * (freqs >= RATE / 4),
        n=LENGTH,
    )
    reconstructed = low + generated_residual

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

torch.save({
    "model": model.state_dict(),
    "residual_scale": residual_scale.item(),
    "rate": RATE,
    "patch": PATCH,
    "patches": PATCHES,
    "width": WIDTH,
    "diffusion_steps": STEPS,
}, out / "denoiser.pt")
print(f"Baseline MSE:       {(low - target).square().mean().item():.9e}")
print(f"Reconstruction MSE: {(reconstructed - target).square().mean().item():.9e}")
print(f"PASS: training and sampling completed; outputs in {out}/")
