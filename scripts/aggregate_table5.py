#!/usr/bin/env python3
"""Aggregate early-exit threshold sweeps into the AVA-VLA Table-5 columns."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TABLE5_THRESHOLDS = (0.30, 0.40, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00)


def aggregate(result_root: Path, expected_thresholds) -> dict:
    output_rows = []
    for threshold in expected_thresholds:
        threshold_dir = result_root / f"threshold_{threshold:.2f}"
        suite_rows = []
        for suite in SUITES:
            path = threshold_dir / suite / "evaluation_results.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Table-5 result: {path}")
            with path.open(encoding="utf-8") as stream:
                row = json.load(stream)
            if row.get("task_suite") != suite:
                raise ValueError(f"Wrong suite in {path}: {row.get('task_suite')}")
            if not np.isclose(float(row.get("exit_threshold", np.nan)), threshold):
                raise ValueError(f"Wrong threshold in {path}: {row.get('exit_threshold')}")
            telemetry = row.get("reasoning_telemetry")
            if not telemetry:
                raise ValueError(f"No reasoning telemetry in {path}")
            suite_rows.append(row)
        checkpoints = {str(row["checkpoint"]) for row in suite_rows}
        if len(checkpoints) != 1:
            raise ValueError(f"Threshold {threshold:.2f} did not use one Table-1 checkpoint: {checkpoints}")
        latency = [
            float(value)
            for row in suite_rows
            for value in row["reasoning_telemetry"]["latency_ms_values"]
        ]
        steps = [
            float(value)
            for row in suite_rows
            for value in row["reasoning_telemetry"]["reasoning_steps_values"]
        ]
        if not latency or len(latency) != len(steps):
            raise ValueError(f"Invalid Table-5 telemetry for threshold {threshold:.2f}")
        output_rows.append(
            {
                "exit_threshold": float(threshold),
                "avg_reasoning_steps": float(np.mean(steps)),
                "mean_latency_ms": float(np.mean(latency)),
                "p90_latency_ms": float(np.percentile(latency, 90)),
                "avg_success_rate_percent": float(
                    100.0 * np.mean([float(row["success_rate"]) for row in suite_rows])
                ),
                "policy_queries": len(latency),
                "checkpoint": next(iter(checkpoints)),
                "success_rate_percent_by_suite": {
                    row["task_suite"]: 100.0 * float(row["success_rate"]) for row in suite_rows
                },
            }
        )
    return {"table": "Table 5", "rows": output_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=TABLE5_THRESHOLDS,
    )
    args = parser.parse_args()
    payload = aggregate(args.result_root, args.thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, args.output)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "exit_threshold", "avg_reasoning_steps", "mean_latency_ms",
                "p90_latency_ms", "avg_success_rate_percent", "policy_queries", "checkpoint",
            ),
        )
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
