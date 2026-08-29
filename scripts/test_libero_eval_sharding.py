"""Regression tests for exact, disjoint LIBERO evaluation sharding."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from merge_libero_eval_shards import merge_shards


def _make_shard(root: Path, index: int, count: int = 8) -> Path:
    trial_indices = list(range(index, 50, count))
    episodes = 10 * len(trial_indices)
    successes = episodes - index
    result = {
        "task_suite": "libero_spatial",
        "total_episodes": episodes,
        "total_successes": successes,
        "success_rate": successes / episodes,
        "seed": 7,
        "checkpoint": "/checkpoint/run",
        "num_tasks": 10,
        "num_trials_per_task": 50,
        "trial_shard_index": index,
        "trial_shard_count": count,
        "trial_indices": trial_indices,
    }
    path = root / f"shard_{index}.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        paths = [_make_shard(root, index) for index in range(8)]
        output = root / "evaluation_results.json"
        merged = merge_shards(paths, output)
        assert merged["total_episodes"] == 500
        assert merged["trial_shard_count"] == 8
        assert output.is_file()
        assert (root / "EVALUATION_COMPLETE").is_file()

        corrupt = json.loads(paths[3].read_text(encoding="utf-8"))
        corrupt["trial_indices"] = corrupt["trial_indices"][:-1]
        paths[3].write_text(json.dumps(corrupt), encoding="utf-8")
        try:
            merge_shards(paths, root / "must_not_exist.json")
        except RuntimeError as error:
            assert "coverage mismatch" in str(error)
        else:
            raise AssertionError("Merger accepted an incomplete evaluation shard")

    eval_source = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "robot"
        / "libero"
        / "run_libero_eval.py"
    ).read_text(encoding="utf-8")
    assert "range(cfg.trial_shard_index, cfg.num_trials_per_task, cfg.trial_shard_count)" in eval_source
    assert "Incomplete shard evaluation" in eval_source
    print("PASS: exact 8-shard coverage, merge, and corruption rejection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
