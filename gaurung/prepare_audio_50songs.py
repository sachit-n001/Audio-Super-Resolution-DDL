import json
import shutil
import wave
import zipfile
from pathlib import Path

import numpy as np

source = Path("data/multiclip_pilot")
destination = Path("data/multiclip_50songs")
archive_path = Path("data/musdb18hq.zip")
rate = 44100
length = 8192

# Never overwrite an existing prepared dataset.
if destination.exists():
    raise RuntimeError(
        f"{destination} already exists. Check it before rerunning."
    )

manifest = json.loads((source / "manifest.json").read_text())
validation_songs = {x["song"] for x in manifest["validation"]}
original_train = sorted({x["song"] for x in manifest["train"]})

clips = []
records = []

with zipfile.ZipFile(archive_path) as archive:
    candidates = sorted(
        name for name in archive.namelist()
        if name.startswith("train/")
        and name.endswith("/mixture.wav")
        and name not in validation_songs
    )

    assert set(original_train).issubset(candidates)
    selected = original_train + [
        name for name in candidates if name not in original_train
    ]
    selected = selected[:50]
    assert len(selected) == 50

    for number, song in enumerate(selected, 1):
        count = 0
        with archive.open(song) as member:
            with wave.open(member, "rb") as audio:
                assert audio.getframerate() == rate
                assert audio.getcomptype() == "NONE"
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                assert width in (2, 3), f"Unsupported PCM width: {width}"

                # Read in increasing order to limit ZIP seeking.
                for offset in range(10, 110, 5):
                    start = offset * rate
                    if start + length > audio.getnframes():
                        continue
                    audio.setpos(start)
                    raw = audio.readframes(length)

                    if width == 2:
                        samples = np.frombuffer(raw, dtype="<i2")
                        samples = samples.astype(np.float32) / 32768.0
                    else:
                        packed = np.frombuffer(raw, dtype=np.uint8)
                        packed = packed.reshape(-1, 3).astype(np.int32)
                        samples = (
                            packed[:, 0]
                            | (packed[:, 1] << 8)
                            | (packed[:, 2] << 16)
                        )
                        samples = (samples ^ 0x800000) - 0x800000
                        samples = samples.astype(np.float32) / 8388608.0

                    # Match the pilot's left-channel convention.
                    clip = samples.reshape(-1, channels)[:, 0].copy()
                    assert clip.shape == (length,)
                    assert np.isfinite(clip).all()
                    clips.append(clip)
                    records.append({
                        "song": song,
                        "offset_seconds": offset,
                    })
                    count += 1

        assert count > 0, f"No usable clips: {song}"
        print(f"{number}/50: {song}: {count} clips", flush=True)

assert {r["song"] for r in records}.isdisjoint(validation_songs)

destination.mkdir()
np.save(destination / "train.npy", np.stack(clips))

# Preserve exactly the same held-out examples for comparison.
shutil.copyfile(
    source / "validation.npy",
    destination / "validation.npy",
)

manifest["train"] = records
manifest["sample_rate"] = rate
manifest["samples_per_clip"] = length
manifest["channel"] = "left"
manifest["preparation"] = "50 training songs; offsets 10..105s every 5s"
(destination / "manifest.json").write_text(
    json.dumps(manifest, indent=2)
)

print(f"PREPARATION COMPLETE: {len(clips)} training clips", flush=True)
