"""Evaluate one AVA-VLA checkpoint on all four LIBERO suites with disjoint GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_libero_eval_shards import merge_shards  # noqa: E402


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _valid_result(path: Path, suite: str, checkpoint: Path, trials: int, shards: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("task_suite") == suite
        and Path(str(result.get("checkpoint", ""))).resolve() == checkpoint.resolve()
        and int(result.get("total_episodes", -1)) == 10 * trials
        and int(result.get("trial_shard_count", -1)) == shards
    )


def evaluate_suite(args, suite: str) -> dict:
    suite_dir = args.output_root / suite
    merged_path = suite_dir / "evaluation_results.json"
    if _valid_result(merged_path, suite, args.checkpoint, args.num_trials_per_task, args.shards):
        return json.loads(merged_path.read_text(encoding="utf-8"))

    processes: List[tuple[int, subprocess.Popen, object]] = []
    shard_paths: List[Path] = []
    for shard in range(args.shards):
        shard_dir = suite_dir / f"shard_{shard}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / "evaluation_results.json"
        shard_paths.append(shard_path)
        log_path = shard_dir / "evaluation.log"
        command = [
            str(args.python),
            str(PROJECT_ROOT / "experiments" / "robot" / "libero" / "run_libero_eval.py"),
            "--model_family", "avavla",
            "--pretrained_checkpoint", str(args.checkpoint),
            "--use_l1_regression", "true",
            "--num_images_in_input", "2",
            "--use_proprio", "true",
            "--enable_latent_reasoning", "true",
            "--max_reasoning_steps", str(args.max_reasoning_steps),
            "--exit_threshold", str(args.exit_threshold),
            "--use_history_state", "true",
            "--center_crop", "true",
            "--num_open_loop_steps", "8",
            "--task_suite_name", suite,
            "--num_steps_wait", "10",
            "--num_trials_per_task", str(args.num_trials_per_task),
            "--trial_shard_index", str(shard),
            "--trial_shard_count", str(args.shards),
            "--initial_states_path", "DEFAULT",
            "--env_img_res", "256",
            "--local_log_dir", str(shard_dir),
            "--save_video", "false",
            "--use_wandb", "false",
            "--seed", str(args.eval_seed),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        log_stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard, process, log_stream))

    failures = []
    for shard, process, log_stream in processes:
        return_code = process.wait()
        log_stream.close()
        if return_code:
            failures.append((shard, return_code))
    if failures:
        raise RuntimeError(f"LIBERO {suite} evaluation shards failed: {failures}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    return merge_shards(shard_paths, merged_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--max-reasoning-steps", type=int, default=5)
    parser.add_argument("--exit-threshold", type=float, default=0.55)
    args = parser.parse_args()
    if not 1 <= args.shards <= 8:
        raise SystemExit("--shards must be in [1, 8]")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {args.checkpoint}")

    summary = {}
    for suite in SUITES:
        result = evaluate_suite(args, suite)
        summary[suite] = {
            "success_rate": result["success_rate"],
            "total_successes": result["total_successes"],
            "total_episodes": result["total_episodes"],
        }
    summary["average_success_rate"] = sum(
        row["success_rate"] for row in summary.values()
    ) / len(SUITES)
    summary["checkpoint"] = str(args.checkpoint)
    output = args.output_root / "all_suites_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
