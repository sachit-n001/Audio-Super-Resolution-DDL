import json
import wave
import zipfile
from pathlib import Path

import numpy as np

destination = Path("data/final_test")
if destination.exists():
    raise RuntimeError("data/final_test already exists; stopped.")

rate = 44100
length = 8192
offsets = [15, 30, 45, 60]
clips = []
records = []

with zipfile.ZipFile("data/musdb18hq.zip") as archive:
    songs = sorted(
        name for name in archive.namelist()
        if name.startswith("test/") and name.endswith("/mixture.wav")
    )
    if len(songs) != 50:
        raise RuntimeError(f"Expected 50 test songs, found {len(songs)}")

    # Check test-song separation from both training manifests.
    used_songs = set()
    for name in ["multiclip_pilot", "multiclip_50songs"]:
        manifest = json.loads(
            (Path("data") / name / "manifest.json").read_text()
        )
        for split in ["train", "validation"]:
            used_songs.update(item["song"] for item in manifest[split])
    assert used_songs.isdisjoint(songs), "Test-song overlap!"

    for song in songs:
        with archive.open(song) as stream:
            with wave.open(stream, "rb") as audio:
                if audio.getframerate() != rate:
                    raise RuntimeError(f"Unexpected sample rate: {song}")
                channels = audio.getnchannels()
                width = audio.getsampwidth()

                for offset in offsets:
                    start = offset * rate
                    if start + length > audio.getnframes():
                        raise RuntimeError(f"Clip exceeds song length: {song}")
                    audio.setpos(start)
                    raw = audio.readframes(length)

                    if width == 2:
                        values = np.frombuffer(raw, dtype="<i2")
                        values = values.astype(np.float32) / 32768
                    elif width == 3:
                        b = np.frombuffer(raw, dtype=np.uint8)
                        b = b.reshape(-1, 3).astype(np.int32)
                        values = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                        values = (values ^ 0x800000) - 0x800000
                        values = values.astype(np.float32) / 8388608
                    else:
                        raise RuntimeError(f"Unsupported PCM width: {width}")

                    clip = values.reshape(-1, channels)[:, 0].copy()
                    assert clip.shape == (length,)
                    assert np.isfinite(clip).all()
                    clips.append(clip)
                    records.append({
                        "song": song,
                        "offset_seconds": offset,
                    })
        print(f"Prepared {song}", flush=True)

array = np.stack(clips)
destination.mkdir(parents=True)
np.save(destination / "test.npy", array)
(destination / "manifest.json").write_text(json.dumps({
    "sample_rate": rate,
    "samples_per_clip": length,
    "channel": "left",
    "offsets_seconds": offsets,
    "test": records,
}, indent=2))
print(f"Saved {len(array)} clips from {len(songs)} test songs.", flush=True)
