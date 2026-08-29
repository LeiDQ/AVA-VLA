"""Aggregate the two AVA-VLA/Ours rows in paper Table 1 from LIBERO result JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEED_PATTERN = re.compile(r"^seed(?P<seed>\d+)$")


def _training_seed(path: Path) -> int:
    for part in path.parts:
        match = SEED_PATTERN.match(part)
        if match:
            return int(match.group("seed"))
    checkpoint_match = re.search(r"seed(?P<seed>\d+)", str(path))
    if checkpoint_match:
        return int(checkpoint_match.group("seed"))
    raise RuntimeError(f"Cannot infer training seed from {path}")


def discover_results(result_root: Path) -> Dict[Tuple[int, str], dict]:
    rows: Dict[Tuple[int, str], dict] = {}
    paths = sorted(result_root.rglob("evaluation_results.json"))
    for path in paths:
        if any(part.startswith("shard_") for part in path.parts):
            continue
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        suite = result.get("task_suite")
        if suite not in SUITES:
            continue
        seed = _training_seed(path.relative_to(result_root))
        episodes = int(result.get("total_episodes", 0))
        successes = int(result.get("total_successes", -1))
        rate = float(result.get("success_rate", math.nan))
        if episodes <= 0 or not (0 <= successes <= episodes):
            raise RuntimeError(f"Invalid episode counts in {path}")
        if not math.isclose(rate, successes / episodes, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Inconsistent success rate in {path}")
        key = (seed, suite)
        if key in rows:
            raise RuntimeError(f"Duplicate result for training seed {seed}, suite {suite}")
        rows[key] = {**result, "source": str(path), "training_seed": seed}
    if not rows:
        raise RuntimeError(f"No merged LIBERO evaluation results found under {result_root}")
    return rows


def aggregate(
    rows: Dict[Tuple[int, str], dict],
    mode: str,
    expected_seeds: Iterable[int],
) -> dict:
    seeds = tuple(expected_seeds)
    missing = [
        (seed, suite) for seed in seeds for suite in SUITES if (seed, suite) not in rows
    ]
    if missing:
        raise RuntimeError(f"Missing Table-1 results: {missing}")

    per_seed = []
    for seed in seeds:
        suite_rows = [rows[(seed, suite)] for suite in SUITES]
        checkpoints = [str(row.get("checkpoint", "")) for row in suite_rows]
        if mode == "all_policy" and len(set(checkpoints)) != 1:
            raise RuntimeError(
                f"All-policy seed {seed} must evaluate one checkpoint on all suites; got {checkpoints}"
            )
        success_rates = {
            suite: 100.0 * float(rows[(seed, suite)]["success_rate"]) for suite in SUITES
        }
        per_seed.append(
            {
                "training_seed": seed,
                "checkpoint_by_suite": dict(zip(SUITES, checkpoints)),
                "success_rate_percent": success_rates,
                "average_percent": mean(success_rates.values()),
            }
        )

    columns = {}
    for suite in SUITES:
        values = [row["success_rate_percent"][suite] for row in per_seed]
        columns[suite] = {
            "mean_percent": mean(values),
            "std_percent": pstdev(values),
            "values_percent": values,
        }
    averages = [row["average_percent"] for row in per_seed]
    return {
        "mode": mode,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "table1_row": {
            "spatial": columns["libero_spatial"],
            "object": columns["libero_object"],
            "goal": columns["libero_goal"],
            "long": columns["libero_10"],
            "average": {
                "mean_percent": mean(averages),
                "std_percent": pstdev(averages),
                "values_percent": averages,
            },
        },
    }


def markdown_summary(summary: dict) -> str:
    row = summary["table1_row"]
    label = (
        "One policy for all 4 suites"
        if summary["mode"] == "all_policy"
        else "One policy per suite"
    )
    columns = [row[name]["mean_percent"] for name in ("spatial", "object", "goal", "long", "average")]
    values = " | ".join(f"{value:.2f}" for value in columns)
    return (
        "| Setting | Spatial | Object | Goal | Long | Average |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| {label} | {values} |\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("all_policy", "per_suite"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate(discover_results(args.result_root), args.mode, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_summary(summary), encoding="utf-8")
    print(markdown_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
