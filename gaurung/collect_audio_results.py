import csv
import json
from pathlib import Path

output = Path("analysis")
output.mkdir(exist_ok=True)

columns = [
    "run", "gpu", "world_size", "training_precision",
    "training_seed", "repeat", "layers", "width",
    "learning_rate", "global_batch", "updates",
    "train_clips", "validation_clips",
    "measured_training_seconds", "mean_step_ms",
    "clips_per_second", "peak_allocated_gib_after_warmup",
    "validation_baseline_mse", "validation_reconstruction_mse",
    "validation_improvement_percent",
]

rows = []
for path in sorted(Path("runs").glob("*/metrics.json")):
    try:
        metrics = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"Skipped unreadable metrics: {path}")
        continue

    row = {key: metrics.get(key, "") for key in columns}
    row["run"] = path.parent.name
    rows.append(row)

destination = output / "completed_training_runs.csv"
with destination.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Collected {len(rows)} metric files into {destination}")
print("Blank fields mean the original run did not record that value.")
print("Check Slurm completion status separately.")
