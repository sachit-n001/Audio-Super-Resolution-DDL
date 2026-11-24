from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(
        self,
        data_dir="data/processed",
        split="train",
        val_fraction=0.1,
        seed=42
    ):
        data_dir = Path(data_dir)

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")

        if split == "test":
            songs = sorted(
                p for p in (data_dir / "test").iterdir()
                if p.is_dir()
            )

        else:
            songs = sorted(
                p for p in (data_dir / "train").iterdir()
                if p.is_dir()
            )

            random.Random(seed).shuffle(songs)

            num_val = max(
                1, int(len(songs) * val_fraction)
            )

            if split == "val":
                songs = songs[:num_val]
            else:
                songs = songs[num_val:]

        self.files = []

        for song in songs:
            self.files.extend(
                sorted(song.glob("segment_*.npz"))
            )

        if not self.files:
            raise RuntimeError(
                f"No audio segments found for {split}"
            )

        print(
            f"{split}: {len(songs)} songs, "
            f"{len(self.files)} segments"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        with np.load(self.files[index]) as data:
            low = torch.from_numpy(data["low"]).float()
            high = torch.from_numpy(data["high"]).float()

        return low, high