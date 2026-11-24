import argparse
import re
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_training_log(path):

    content = path.read_text()

    pattern = (
        r"Epoch (\d+)/\d+ \| "
        r"Train loss: ([\d.]+) \| "
        r"Val loss: ([\d.]+) \| "
        r"Time: ([\d.]+)s \| "
        r"Throughput: ([\d.]+) samples/s"
    )

    matches = re.findall(pattern, content)

    if not matches:
        raise ValueError(f"No epoch results found in {path}")

    epochs = []
    train_loss = []
    val_loss = []
    times = []
    throughput = []

    for match in matches:

        epochs.append(int(match[0]))
        train_loss.append(float(match[1]))
        val_loss.append(float(match[2]))
        times.append(float(match[3]))
        throughput.append(float(match[4]))

    steps = re.findall(r"Total steps: (\d+)", content)
    samples = re.findall(r"Total samples: (\d+)", content)
    average = re.findall(
        r"Average throughput: ([\d.]+)", content
    )

    if not steps or not samples or not average:
        raise ValueError(f"Incomplete training run: {path}")

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "training_time": sum(times),
        "throughput": float(average[-1]),
        "steps": int(steps[-1]),
        "samples": int(samples[-1])
    }


def parse_gpu_utilization(paths):

    values = []

    for path in paths:

        with path.open() as file:

            reader = csv.DictReader(file)

            for row in reader:

                for key, value in row.items():

                    if key and key.strip().startswith(
                        "utilization.gpu"
                    ):

                        try:
                            values.append(
                                float(value.replace("%", "").strip())
                            )
                        except (ValueError, AttributeError):
                            continue

    return np.mean(values) if values else np.nan


def save_bar_chart(
    gpu_counts, values, title, ylabel, filename, output_dir
):

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        [f"{n} GPU" if n == 1 else f"{n} GPUs"
         for n in gpu_counts],
        values
    )

    for bar, value in zip(bars, values):

        if np.isfinite(value):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom"
            )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.margins(y=0.15)
    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default="logs")

    parser.add_argument("--job_1gpu", required=True)
    parser.add_argument("--job_4gpu", required=True)
    parser.add_argument("--job_8gpu", required=True)

    parser.add_argument(
        "--out_dir",
        default="outputs/performance"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        1: args.job_1gpu,
        4: args.job_4gpu,
        8: args.job_8gpu
    }

    results = {}

    for gpu_count, job_id in jobs.items():

        log_path = (
            log_dir / f"train_{gpu_count}gpu_{job_id}.out"
        )

        results[gpu_count] = parse_training_log(log_path)

        gpu_logs = list(
            log_dir.glob(
                f"gpu_{gpu_count}gpu_{job_id}.csv"
            )
        )

        gpu_logs += list(
            log_dir.glob(
                f"gpu_{gpu_count}gpu_{job_id}_*.csv"
            )
        )

        results[gpu_count]["utilization"] = (
            parse_gpu_utilization(gpu_logs)
        )

    gpu_counts = [1, 4, 8]

    # Verify comparable training workloads
    workloads = {
        (results[n]["steps"], results[n]["samples"])
        for n in gpu_counts
    }

    if len(workloads) != 1:
        raise ValueError(
            "Training runs have different workloads. "
            "Use matching steps and sample counts."
        )

    baseline_throughput = results[1]["throughput"]

    for n in gpu_counts:

        results[n]["speedup"] = (
            results[n]["throughput"] / baseline_throughput
        )

        results[n]["efficiency"] = (
            results[n]["speedup"] / n * 100
        )

    charts = [
        (
            "training_time",
            "Training Time Comparison",
            "Time (seconds)",
            "training_time.png"
        ),
        (
            "throughput",
            "Training Throughput",
            "Samples per second",
            "throughput.png"
        ),
        (
            "utilization",
            "Average Logged GPU Utilization",
            "GPU utilization (%)",
            "gpu_utilization.png"
        ),
        (
            "speedup",
            "Distributed Training Speedup",
            "Speedup (x)",
            "speedup.png"
        ),
        (
            "efficiency",
            "Distributed Scaling Efficiency",
            "Efficiency (%)",
            "scaling_efficiency.png"
        )
    ]

    for metric, title, ylabel, filename in charts:

        save_bar_chart(
            gpu_counts,
            [results[n][metric] for n in gpu_counts],
            title,
            ylabel,
            filename,
            output_dir
        )

    # Training and validation loss curves
    for loss_type in ["train_loss", "val_loss"]:

        plt.figure(figsize=(9, 5))

        for n in gpu_counts:

            plt.plot(
                results[n]["epochs"],
                results[n][loss_type],
                marker="o",
                label=f"{n} GPU(s)"
            )

        plt.title(
            "Training Loss"
            if loss_type == "train_loss"
            else "Validation Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{loss_type}.png",
            dpi=200
        )

        plt.close()

    # Export numerical results
    summary_path = output_dir / "performance_summary.csv"

    with summary_path.open("w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "GPU Count",
            "Training Steps",
            "Training Samples",
            "Training Time (s)",
            "Throughput (samples/s)",
            "Logged GPU Utilization (%)",
            "Speedup",
            "Scaling Efficiency (%)",
            "Final Training Loss",
            "Final Validation Loss"
        ])

        for n in gpu_counts:

            r = results[n]

            writer.writerow([
                n,
                r["steps"],
                r["samples"],
                r["training_time"],
                r["throughput"],
                r["utilization"],
                r["speedup"],
                r["efficiency"],
                r["train_loss"][-1],
                r["val_loss"][-1]
            ])

    print("\nPerformance Summary")

    for n in gpu_counts:

        r = results[n]

        print(
            f"{n} GPU(s) | "
            f"Time: {r['training_time']:.2f}s | "
            f"Throughput: {r['throughput']:.2f} | "
            f"Utilization: {r['utilization']:.2f}% | "
            f"Speedup: {r['speedup']:.2f}x | "
            f"Efficiency: {r['efficiency']:.2f}%"
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()