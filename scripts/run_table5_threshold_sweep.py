#!/usr/bin/env python3
"""Run the Table-5 early-exit sweep using one Table-1 all-suites checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_table5 import TABLE5_THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=TABLE5_THRESHOLDS,
    )
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    for threshold in args.thresholds:
        output = args.output_root / f"threshold_{threshold:.2f}"
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_libero_checkpoint_all_suites.py"),
            "--checkpoint", str(args.checkpoint),
            "--output-root", str(output),
            "--shards", str(args.shards),
            "--num-trials-per-task", str(args.num_trials_per_task),
            "--exit-threshold", str(threshold),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/aggregate_table5.py"),
            "--result-root", str(args.output_root),
            "--output", str(args.output_root / "table5_results.json"),
            "--thresholds", *[str(value) for value in args.thresholds],
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
