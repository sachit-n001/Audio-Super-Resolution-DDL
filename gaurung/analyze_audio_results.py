import csv
import json
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path("analysis/audio_examples")
output.mkdir(parents=True, exist_ok=True)

def read_audio(path):
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        rate = audio.getframerate()
        samples = np.frombuffer(
            audio.readframes(audio.getnframes()), dtype="<i2"
        ).astype(np.float64) / 32768.0
    return samples, rate

def spectrum(samples):
    frames = np.lib.stride_tricks.sliding_window_view(
        samples, 1024
    )[::256]
    return np.abs(
        np.fft.rfft(frames * np.hanning(1024), axis=-1)
    ).T

def evaluate(reference, estimate):
    target = spectrum(reference)
    predicted = spectrum(estimate)

    # One fixed magnitude floor for every comparison.
    floor = 1e-7
    difference_db = (
        20 * np.log10(np.maximum(predicted, floor))
        - 20 * np.log10(np.maximum(target, floor))
    )
    return {
        "mse": float(np.mean((estimate - reference) ** 2)),
        "log_spectral_distance_db": float(
            np.mean(np.sqrt(np.mean(difference_db ** 2, axis=0)))
        ),
        "spectral_convergence": float(
            np.linalg.norm(predicted - target)
            / max(np.linalg.norm(target), 1e-12)
        ),
    }

rows = []

for target_path in sorted(Path("runs").glob("*/validation_target.wav")):
    run = target_path.parent
    degraded_path = run / "validation_degraded.wav"
    reconstructed_path = run / "validation_reconstructed.wav"
    if not degraded_path.exists() or not reconstructed_path.exists():
        continue

    target, rate = read_audio(target_path)
    degraded, degraded_rate = read_audio(degraded_path)
    reconstructed, reconstructed_rate = read_audio(reconstructed_path)

    assert rate == degraded_rate == reconstructed_rate
    assert target.shape == degraded.shape == reconstructed.shape

    for label, samples in (
        ("degraded", degraded),
        ("reconstructed", reconstructed),
    ):
        rows.append({
            "run": run.name,
            "scope": "first_saved_validation_clip_PCM16",
            "signal": label,
            **evaluate(target, samples),
        })

    signals = [target, degraded, reconstructed]
    magnitudes = [
        20 * np.log10(np.maximum(spectrum(x), 1e-7))
        for x in signals
    ]
    maximum = max(float(x.max()) for x in magnitudes)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label, magnitude in zip(
        axes, ("Target", "Degraded", "Reconstructed"), magnitudes
    ):
        axis.imshow(
            magnitude,
            origin="lower",
            aspect="auto",
            extent=(0, len(target) / rate, 0, rate / 2000),
            vmin=maximum - 80,
            vmax=maximum,
            cmap="magma",
        )
        axis.axhline(rate / 4000, color="cyan", linestyle="--")
        axis.set_title(label)
        axis.set_ylabel("Frequency (kHz)")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(output / f"{run.name}_spectrogram.png", dpi=160)
    plt.close(fig)

if not rows:
    raise RuntimeError("No complete validation WAV examples found.")

with (output / "example_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(output / "method.json").write_text(json.dumps({
    "scope": "First saved validation clip per run, not the full validation set",
    "audio": "Saved PCM16 WAV; clipping and quantization may affect metrics",
    "fft_size": 1024,
    "hop_samples": 256,
    "window": "numpy.hanning",
    "magnitude_floor": 1e-7,
    "log_spectral_distance": "Mean over frames of RMS magnitude-dB error across bins",
    "comparison": "Compare runs only when their target clips and tasks match",
}, indent=2))

print(f"Analyzed {len(rows) // 2} runs.")
print(f"Saved metrics and spectrograms to {output}")
