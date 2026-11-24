import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Audio configuration
HIGH_SR = 44100
LOW_SR = 8000
SEGMENT_SECONDS = 2

HIGH_SAMPLES = HIGH_SR * SEGMENT_SECONDS
LOW_SAMPLES = LOW_SR * SEGMENT_SECONDS


def process_song(audio_path, output_dir):
    # Load original high-resolution audio
    audio, sample_rate = sf.read(
        audio_path, dtype="float32", always_2d=True
    )

    if sample_rate != HIGH_SR:
        raise ValueError(f"Unexpected sample rate: {sample_rate}")

    # Convert stereo audio to mono
    audio = audio.mean(axis=1)

    # Downsample from 44.1 kHz to 8 kHz
    low_audio = resample_poly(audio, 80, 441)

    song_name = audio_path.parent.name
    song_dir = output_dir / song_name
    song_dir.mkdir(parents=True, exist_ok=True)

    num_segments = min(
        len(audio) // HIGH_SAMPLES,
        len(low_audio) // LOW_SAMPLES
    )

    # Generate matching low/high-resolution segments
    for i in range(num_segments):
        high_segment = audio[
            i * HIGH_SAMPLES:(i + 1) * HIGH_SAMPLES
        ]

        low_segment = low_audio[
            i * LOW_SAMPLES:(i + 1) * LOW_SAMPLES
        ]

        np.savez(
            song_dir / f"segment_{i:05d}.npz",
            low=low_segment.astype(np.float32),
            high=high_segment.astype(np.float32)
        )

    print(f"{song_name}: {num_segments} segments")
    return num_segments


def prepare_dataset(dataset_dir, output_dir):
    for split in ["train", "test"]:
        songs = sorted(
            (dataset_dir / split).glob("*/mixture.wav")
        )

        if not songs:
            raise FileNotFoundError(
                f"No mixture.wav files found in {dataset_dir / split}"
            )

        total_segments = 0

        print(f"\nProcessing {split}: {len(songs)} songs")

        for song in songs:
            total_segments += process_song(
                song, output_dir / split
            )

        print(f"{split}: {total_segments} total segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/musdb18hq")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed")
    )

    args = parser.parse_args()

    prepare_dataset(args.dataset_dir, args.output_dir)