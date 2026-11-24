import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import stft
from models.audio_transformer import GPT, GPTConfig
from scripts.audio_dataset import AudioDataset


def spectral_distance(reference, prediction, sr):

    _, _, ref_stft = stft(
        reference, fs=sr, nperseg=1024, noverlap=768
    )

    _, _, pred_stft = stft(
        prediction, fs=sr, nperseg=1024, noverlap=768
    )

    ref_db = 20 * np.log10(
        np.maximum(np.abs(ref_stft), 1e-7)
    )

    pred_db = 20 * np.log10(
        np.maximum(np.abs(pred_stft), 1e-7)
    )

    return np.sqrt(np.mean((ref_db - pred_db) ** 2))


def plot_spectrograms(high, baseline, reconstructed, sr, path):

    signals = [
        ("Original 44.1 kHz Audio", high),
        ("Interpolated 8 kHz Audio", baseline),
        ("AI-Reconstructed 44.1 kHz Audio", reconstructed)
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True
    )

    for ax, (title, audio) in zip(axes, signals):

        frequencies, times, spectrum = stft(
            audio,
            fs=sr,
            nperseg=1024,
            noverlap=768
        )

        magnitude = 20 * np.log10(
            np.maximum(np.abs(spectrum), 1e-7)
        )

        image = ax.pcolormesh(
            times,
            frequencies,
            magnitude,
            shading="auto",
            cmap="magma",
            vmin=-100,
            vmax=0
        )

        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, sr / 2)

        ax.axhline(
            4000,
            color="white",
            linestyle="--",
            linewidth=0.8
        )

        fig.colorbar(image, ax=ax, label="Magnitude (dB)")

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/1gpu/best.pt"
    )

    parser.add_argument(
        "--data_dir",
        default="data/processed"
    )

    parser.add_argument(
        "--out_dir",
        default="outputs/evaluation"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Evaluation device: {device}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    config = GPTConfig(**checkpoint["config"])

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load held-out test dataset
    dataset = AudioDataset(
        data_dir=args.data_dir,
        split="test"
    )

    num_samples = min(args.num_samples, len(dataset))

    if num_samples <= 0:
        raise ValueError("Number of samples must be positive")

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(dataset),
        size=num_samples,
        replace=False
    )

    baseline_mae = []
    model_mae = []

    baseline_spectral = []
    model_spectral = []

    with torch.no_grad():

        for i, index in enumerate(indices):

            low, high = dataset[int(index)]

            low = low.unsqueeze(0).to(device)
            high = high.unsqueeze(0).to(device)

            reconstructed, _ = model(low)

            baseline = F.interpolate(
                low.unsqueeze(1),
                size=high.shape[-1],
                mode="linear",
                align_corners=False
            ).squeeze(1)

            baseline_mae.append(
                F.l1_loss(baseline, high).item()
            )

            model_mae.append(
                F.l1_loss(reconstructed, high).item()
            )

            low_np = low.squeeze().cpu().numpy()
            high_np = high.squeeze().cpu().numpy()
            baseline_np = baseline.squeeze().cpu().numpy()

            reconstructed_np = (
                reconstructed.squeeze().cpu().numpy()
            )

            baseline_spectral.append(
                spectral_distance(
                    high_np,
                    baseline_np,
                    config.high_sr
                )
            )

            model_spectral.append(
                spectral_distance(
                    high_np,
                    reconstructed_np,
                    config.high_sr
                )
            )

            # Save first three audio examples
            if i < 3:

                sample_dir = output_dir / f"sample_{i + 1}"

                sample_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                sf.write(
                    sample_dir / "original.wav",
                    high_np,
                    config.high_sr,
                    subtype="FLOAT"
                )

                sf.write(
                    sample_dir / "low_resolution.wav",
                    low_np,
                    config.low_sr,
                    subtype="FLOAT"
                )

                sf.write(
                    sample_dir / "interpolated.wav",
                    baseline_np,
                    config.high_sr,
                    subtype="FLOAT"
                )

                sf.write(
                    sample_dir / "reconstructed.wav",
                    reconstructed_np,
                    config.high_sr,
                    subtype="FLOAT"
                )

                plot_spectrograms(
                    high_np,
                    baseline_np,
                    reconstructed_np,
                    config.high_sr,
                    sample_dir / "spectrograms.png"
                )

    results = (
        f"Evaluated samples: {num_samples}\n"
        f"Baseline MAE: {np.mean(baseline_mae):.6f}\n"
        f"Model MAE: {np.mean(model_mae):.6f}\n"
        f"Baseline spectral distance: "
        f"{np.mean(baseline_spectral):.4f} dB\n"
        f"Model spectral distance: "
        f"{np.mean(model_spectral):.4f} dB\n"
    )

    print("\nEvaluation Results")
    print(results)

    (output_dir / "results.txt").write_text(results)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()