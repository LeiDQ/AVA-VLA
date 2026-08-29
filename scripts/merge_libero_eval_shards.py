"""Validate and atomically merge disjoint LIBERO evaluation shards."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    required = {
        "task_suite",
        "total_episodes",
        "total_successes",
        "success_rate",
        "seed",
        "checkpoint",
        "num_tasks",
        "num_trials_per_task",
        "trial_shard_index",
        "trial_shard_count",
        "trial_indices",
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"Shard {path} is missing fields: {sorted(missing)}")
    return result


def merge_shards(paths: list[Path], output_path: Path) -> dict:
    if not paths:
        raise RuntimeError("No evaluation shard results were provided")
    shards = [_load_result(path) for path in paths]
    first = shards[0]
    shard_count = int(first["trial_shard_count"])
    num_trials = int(first["num_trials_per_task"])
    num_tasks = int(first["num_tasks"])
    if shard_count != len(shards):
        raise RuntimeError(f"Expected {shard_count} shards, received {len(shards)}")

    invariant_fields = (
        "task_suite",
        "seed",
        "checkpoint",
        "num_tasks",
        "num_trials_per_task",
        "trial_shard_count",
    )
    by_index: dict[int, dict] = {}
    for path, shard in zip(paths, shards):
        for field in invariant_fields:
            if shard[field] != first[field]:
                raise RuntimeError(f"Shard {path} disagrees on {field}")
        index = int(shard["trial_shard_index"])
        if index in by_index or not 0 <= index < shard_count:
            raise RuntimeError(f"Invalid or duplicate shard index {index} in {path}")
        expected_indices = list(range(index, num_trials, shard_count))
        actual_indices = [int(value) for value in shard["trial_indices"]]
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"Shard {index} trial coverage mismatch: {actual_indices} != {expected_indices}"
            )
        expected_episodes = num_tasks * len(expected_indices)
        episodes = int(shard["total_episodes"])
        successes = int(shard["total_successes"])
        if episodes != expected_episodes or not 0 <= successes <= episodes:
            raise RuntimeError(
                f"Shard {index} has invalid counts: successes={successes}, episodes={episodes}, "
                f"expected_episodes={expected_episodes}"
            )
        expected_rate = successes / episodes
        if not math.isclose(float(shard["success_rate"]), expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Shard {index} success_rate does not match its integer counts")
        by_index[index] = shard

    if set(by_index) != set(range(shard_count)):
        raise RuntimeError(f"Incomplete shard set: {sorted(by_index)}")
    covered_trials = sorted(index for shard in by_index.values() for index in shard["trial_indices"])
    if covered_trials != list(range(num_trials)):
        raise RuntimeError("Evaluation shards overlap or leave gaps in trial coverage")

    total_episodes = sum(int(shard["total_episodes"]) for shard in by_index.values())
    total_successes = sum(int(shard["total_successes"]) for shard in by_index.values())
    expected_total = num_tasks * num_trials
    if total_episodes != expected_total:
        raise RuntimeError(f"Merged evaluation has {total_episodes} episodes; expected {expected_total}")

    telemetry_rows = [shard.get("reasoning_telemetry") for shard in by_index.values()]
    if any(row is not None for row in telemetry_rows) and not all(row is not None for row in telemetry_rows):
        raise RuntimeError("Only some LIBERO shards contain reasoning telemetry")
    merged_telemetry = None
    if telemetry_rows and all(row is not None for row in telemetry_rows):
        latency_values = [
            float(value)
            for row in telemetry_rows
            for value in row.get("latency_ms_values", [])
        ]
        step_values = [
            float(value)
            for row in telemetry_rows
            for value in row.get("reasoning_steps_values", [])
        ]
        if len(latency_values) != len(step_values):
            raise RuntimeError("LIBERO telemetry latency and reasoning-step counts disagree")
        merged_telemetry = {
            "policy_queries": len(latency_values),
            "mean_latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
            "p90_latency_ms": float(np.percentile(latency_values, 90)) if latency_values else 0.0,
            "avg_reasoning_steps": float(np.mean(step_values)) if step_values else 0.0,
            "latency_ms_values": latency_values,
            "reasoning_steps_values": step_values,
        }

    merged = {
        "task_suite": first["task_suite"],
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "seed": first["seed"],
        "checkpoint": first["checkpoint"],
        "num_tasks": num_tasks,
        "num_trials_per_task": num_trials,
        "trial_shard_count": shard_count,
        "shards": [by_index[index] for index in range(shard_count)],
    }
    if merged_telemetry is not None:
        merged["reasoning_telemetry"] = merged_telemetry
        for field in ("exit_threshold", "max_reasoning_steps", "fixed_reasoning_steps"):
            values = [shard.get(field) for shard in by_index.values()]
            if len({json.dumps(value, sort_keys=True) for value in values}) != 1:
                raise RuntimeError(f"LIBERO telemetry shards disagree on {field}")
            merged[field] = values[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, output_path)
    complete_path = output_path.parent / "EVALUATION_COMPLETE"
    complete_path.write_text(json.dumps(merged, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    merged = merge_shards(args.shards, args.output)
    print(
        f"MERGED {merged['task_suite']} {merged['total_successes']}/"
        f"{merged['total_episodes']} ({100.0 * merged['success_rate']:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
