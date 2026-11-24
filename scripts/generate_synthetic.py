from pathlib import Path
import numpy as np
import soundfile as sf

sr = 44100
t = np.arange(sr * 6) / sr

audio = (
    0.4 * np.sin(2 * np.pi * 440 * t)
    + 0.2 * np.sin(2 * np.pi * 6000 * t)
    + 0.1 * np.sin(2 * np.pi * 12000 * t)
)

stereo = np.column_stack((audio, audio))

for split in ["train", "test"]:
    folder = Path("data/synthetic") / split / "test_song"
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / "mixture.wav", stereo, sr)

print("Synthetic test audio created successfully")