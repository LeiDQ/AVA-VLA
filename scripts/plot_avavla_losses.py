"""Render every structured AVA-VLA loss series without modifying its values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXTRA_OBJECTIVES = {"entropy_penalty", "mean_composite_reward"}


def ema(values: list[float], smoothing: float) -> list[float]:
    if not values:
        return []
    output = [values[0]]
    for value in values[1:]:
        output.append(smoothing * output[-1] + (1.0 - smoothing) * value)
    return output


def atomic_replace(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def render(metrics_path: Path, output_dir: Path, smoothing: float) -> dict:
    series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with metrics_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {metrics_path}:{line_number}") from exc
            stage = str(row.get("stage", "unknown"))
            step = int(row["global_step"])
            timestamp = float(row.get("timestamp_unix", 0.0))
            for metric, value in row.get("metrics", {}).items():
                if "loss" not in metric.lower() and metric not in EXTRA_OBJECTIVES:
                    continue
                value = float(value)
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite {stage}/{metric} at global step {step}")
                series[f"{stage}/{metric}"].append((step, timestamp, value))
    if not series:
        raise RuntimeError(f"no loss metrics found in {metrics_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss_curves.csv"
    csv_tmp = output_dir / ".loss_curves.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_step", "timestamp_unix", "stage", "metric", "value"])
        for name in sorted(series):
            stage, metric = name.split("/", 1)
            for step, timestamp, value in series[name]:
                writer.writerow([step, timestamp, stage, metric, value])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(csv_tmp, csv_path)

    count = len(series)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 3.8 * rows), squeeze=False)
    for axis, name in zip(axes.flat, sorted(series)):
        points = series[name]
        steps = [point[0] for point in points]
        values = [point[2] for point in points]
        axis.plot(steps, values, color="#4C78A8", alpha=0.22, linewidth=0.7, label="raw")
        axis.plot(steps, ema(values, smoothing), color="#E45756", linewidth=1.6, label="EMA")
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("global step")
        axis.set_ylabel("value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    figure.suptitle(f"AVA-VLA structured losses — {metrics_path.parent.name}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    png_path = output_dir / "loss_curves.png"
    png_tmp = output_dir / ".loss_curves.tmp.png"
    figure.savefig(png_tmp, dpi=160, bbox_inches="tight")
    plt.close(figure)
    atomic_replace(png_tmp, png_path)

    summary = {
        name: {
            "points": len(points),
            "first_step": points[0][0],
            "last_step": points[-1][0],
            "first_value": points[0][2],
            "last_value": points[-1][2],
            "minimum": min(point[2] for point in points),
        }
        for name, points in sorted(series.items())
    }
    json_path = output_dir / "loss_curves_summary.json"
    json_tmp = output_dir / ".loss_curves_summary.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(json_tmp, json_path)
    return {"series": len(series), "png": str(png_path), "csv": str(csv_path), "summary": str(json_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 <= args.smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    result = render(args.metrics.resolve(), args.output_dir.resolve(), args.smoothing)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
