#!/usr/bin/env python3
"""Merge disjoint native CALVIN evaluation shards into Table-3 metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.calvin.protocol import chain_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.shard_dir.glob("shard-*.json"))
    if not paths:
        raise FileNotFoundError(f"No shard JSON files in {args.shard_dir}")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.append(json.load(stream))
    shard_count = int(rows[0]["shard_count"])
    if len(rows) != shard_count or {int(row["shard_index"]) for row in rows} != set(range(shard_count)):
        raise ValueError("CALVIN evaluation shard set is incomplete")
    identity = (rows[0]["checkpoint"], int(rows[0]["num_sequences"]))
    if any((row["checkpoint"], int(row["num_sequences"])) != identity for row in rows):
        raise ValueError("CALVIN shards do not describe the same checkpoint/protocol")
    indexed = {}
    latency_sum = 0.0
    reasoning_sum = 0.0
    query_count = 0
    for row in rows:
        indices = list(map(int, row["sequence_indices"]))
        results = list(map(int, row["sequence_results"]))
        if len(indices) != len(results):
            raise ValueError("CALVIN shard index/result lengths differ")
        for index, result in zip(indices, results):
            if index in indexed:
                raise ValueError(f"Duplicate CALVIN sequence index: {index}")
            indexed[index] = result
        count = int(row.get("policy_queries", 0))
        query_count += count
        latency_sum += float(row.get("mean_latency_ms", 0.0)) * count
        reasoning_sum += float(row.get("avg_reasoning_steps", 0.0)) * count
    expected = set(range(identity[1]))
    if set(indexed) != expected:
        raise ValueError(f"CALVIN shards cover {len(indexed)} sequences, expected {len(expected)}")
    results = [indexed[index] for index in range(identity[1])]
    payload = {
        "benchmark": "calvin_abc_to_d",
        "checkpoint": identity[0],
        "num_sequences": identity[1],
        "shard_count": shard_count,
        "sequence_results": results,
        **chain_metrics(results),
        "policy_queries": query_count,
        "mean_latency_ms": latency_sum / query_count if query_count else 0.0,
        "avg_reasoning_steps": reasoning_sum / query_count if query_count else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
