#!/usr/bin/env python3
"""Fast contract tests for Table 1, Table 3, and Table 5 additions."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.calvin.dataset import CalvinDiskDataset
from experiments.robot.calvin.protocol import _fnv1_32, chain_metrics
from scripts.aggregate_table1_ours import SUITES, aggregate as aggregate_table1
from scripts.aggregate_table5 import TABLE5_THRESHOLDS, aggregate as aggregate_table5
from scripts.merge_libero_eval_shards import merge_shards


def _make_calvin_split(root: Path, split: str) -> None:
    split_dir = root / split
    annotations = split_dir / "lang_annotations"
    annotations.mkdir(parents=True)
    np.save(
        annotations / "auto_lang_ann.npy",
        {
            "info": {"indx": np.asarray([[0, 11]], dtype=np.int64)},
            "language": {
                "ann": np.asarray(["open the drawer"]),
                "task": np.asarray(["open_drawer"]),
                "emb": np.zeros((1, 4), dtype=np.float32),
            },
        },
        allow_pickle=True,
    )
    for index in range(12):
        action = np.linspace(-0.5, 0.5, 7, dtype=np.float32) + index * 0.01
        action[-1] = -1.0 if index % 2 else 1.0
        robot = np.arange(15, dtype=np.float32) * 0.01 + index * 0.001
        robot[-1] = action[-1]
        np.savez_compressed(
            split_dir / f"episode_{index:07d}.npz",
            rel_actions=action,
            actions=action,
            robot_obs=robot,
            scene_obs=np.zeros(24, dtype=np.float32),
            rgb_static=np.full((16, 16, 3), index, dtype=np.uint8),
            rgb_gripper=np.full((8, 8, 3), index, dtype=np.uint8),
        )


def test_calvin_dataset(root: Path) -> None:
    _make_calvin_split(root, "training")
    _make_calvin_split(root, "validation")
    dataset = CalvinDiskDataset(
        root,
        "calvin_abc",
        batch_transform=lambda row: row,
        window_size=9,
        train=True,
        seed=3,
    )
    sample = dataset.raw_sample(0, 0)
    assert sample["action"].shape == (8, 7)
    assert sample["observation"]["image_primary"].shape == (9, 16, 16, 3)
    assert sample["observation"]["image_wrist"].shape == (9, 8, 8, 3)
    assert sample["observation"]["proprio"].shape == (9, 8)
    assert sample["observation"]["pad_mask"].tolist() == [False] * 8 + [True]
    assert set(np.unique(sample["action"][:, -1])).issubset({-1.0, 1.0})
    stats = dataset.dataset_statistics["calvin_abc"]
    assert stats["action"]["mask"] == [True] * 6 + [False]
    assert len(stats["proprio"]["q01"]) == 8


def test_table1() -> None:
    rows = {}
    for seed in (0, 1, 2):
        for suite_index, suite in enumerate(SUITES):
            successes = 350 + 10 * seed + suite_index
            rows[(seed, suite)] = {
                "checkpoint": f"checkpoint-seed{seed}",
                "success_rate": successes / 500.0,
            }
    summary = aggregate_table1(rows, "all_policy", (0, 1, 2))
    assert summary["mode"] == "all_policy"
    assert len(summary["per_seed"]) == 3


def _libero_result(suite: str, shard: int, threshold: float) -> dict:
    values = [1.0 + shard, 2.0 + shard]
    return {
        "task_suite": suite,
        "total_episodes": 10,
        "total_successes": 5 + shard,
        "success_rate": (5 + shard) / 10,
        "seed": 7,
        "checkpoint": "same-checkpoint",
        "num_tasks": 10,
        "num_trials_per_task": 2,
        "trial_shard_index": shard,
        "trial_shard_count": 2,
        "trial_indices": [shard],
        "exit_threshold": threshold,
        "max_reasoning_steps": 5,
        "fixed_reasoning_steps": None,
        "reasoning_telemetry": {
            "policy_queries": 2,
            "mean_latency_ms": float(np.mean(values)),
            "p90_latency_ms": float(np.percentile(values, 90)),
            "avg_reasoning_steps": 2.5 + shard,
            "latency_ms_values": values,
            "reasoning_steps_values": [2.0 + shard, 3.0 + shard],
        },
    }


def test_table5(root: Path) -> None:
    assert TABLE5_THRESHOLDS == (0.30, 0.40, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00)
    threshold = 0.55
    threshold_dir = root / f"threshold_{threshold:.2f}"
    for suite in SUITES:
        shard_paths = []
        for shard in range(2):
            path = threshold_dir / suite / f"shard_{shard}" / "evaluation_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_libero_result(suite, shard, threshold)), encoding="utf-8")
            shard_paths.append(path)
        merge_shards(shard_paths, threshold_dir / suite / "evaluation_results.json")
    summary = aggregate_table5(root, [threshold])
    row = summary["rows"][0]
    assert row["exit_threshold"] == threshold
    assert row["policy_queries"] == 16
    assert row["avg_reasoning_steps"] == 3.0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="avavla-paper-tests-") as temporary:
        root = Path(temporary)
        test_calvin_dataset(root / "calvin_debug_dataset")
        test_table1()
        test_table5(root / "table5")
    metrics = chain_metrics([0, 1, 2, 3, 4, 5])
    assert _fnv1_32("hello world") == 2805756500
    assert np.isclose(metrics["avg_sequence_length"], 2.5)
    assert np.isclose(metrics["success_rate_5"], 1 / 6)
    print("PASS: Table 1 aggregation, CALVIN adapter/protocol, and Table 5 telemetry")


if __name__ == "__main__":
    main()
