#!/usr/bin/env python3
"""Native AVA-VLA evaluator for the official CALVIN ABC->D long-horizon protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robot.calvin.calvin_utils import (
    CALVIN_DATASET_NAME,
    CALVIN_CAMERA_OBS_SPACE,
    add_official_calvin_to_path,
    calvin_images,
    calvin_proprio,
    process_calvin_action,
    resolve_calvin_dataset_root,
)
from experiments.robot.calvin.protocol import chain_metrics, initial_condition_state, official_sequences
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from vla_scripts.deploy_avavla import load_avavla_model, predict_action


def _atomic_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _reasoning_steps(info: Dict) -> float:
    value = info.get("num_steps_performed", 0)
    array = np.asarray(value, dtype=np.float32)
    return float(array.mean()) if array.size else 0.0


def evaluate(args) -> Dict:
    sequences = official_sequences(args.num_sequences, num_workers=args.sequence_workers)
    selected_indices = list(range(args.shard_index, len(sequences), args.shard_count))
    if args.protocol_only:
        return {
            "benchmark": "calvin_abc_to_d",
            "protocol_only": True,
            "num_sequences": len(sequences),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "sequence_indices": selected_indices,
            "first_sequence": list(sequences[0][1]) if sequences else [],
        }

    add_official_calvin_to_path()
    try:
        import hydra
        from omegaconf import OmegaConf
        from calvin_env.envs.play_table_env import get_env
    except ImportError as error:
        raise RuntimeError(
            "CALVIN simulator dependencies are missing. Run scripts/install_calvin_runtime.sh."
        ) from error

    dataset_root = resolve_calvin_dataset_root(args.dataset_root)
    config_root = Path(__file__).resolve().parents[3] / "third_party" / "calvin" / "calvin_models" / "conf"
    task_config = OmegaConf.load(config_root / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_config)
    annotations = OmegaConf.load(config_root / "annotations/new_playtable_validation.yaml")
    env = get_env(
        dataset_root / "validation",
        obs_space=CALVIN_CAMERA_OBS_SPACE,
        show_gui=False,
    )
    model, _, _, _ = load_avavla_model(
        checkpoint_path=args.checkpoint,
        device=args.device,
        enable_latent_reasoning=True,
        max_reasoning_steps=args.max_reasoning_steps,
        exit_threshold=args.exit_threshold,
    )
    if CALVIN_DATASET_NAME not in model.norm_stats:
        raise KeyError(
            f"Checkpoint has no {CALVIN_DATASET_NAME!r} statistics; found {tuple(model.norm_stats)}"
        )

    results: List[int] = []
    latencies_ms: List[float] = []
    reasoning_steps: List[float] = []
    try:
        for sequence_index in tqdm(selected_indices, desc="CALVIN sequences"):
            initial_condition, task_sequence = sequences[sequence_index]
            robot_obs, scene_obs = initial_condition_state(initial_condition)
            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            completed = 0
            for task in task_sequence:
                instruction = str(annotations[task][0])
                if hasattr(model, "reset_latent_history"):
                    model.reset_latent_history()
                queue = deque()
                start_info = env.get_info()
                success = False
                for _ in range(args.episode_length):
                    if not queue:
                        primary, wrist = calvin_images(obs)
                        if torch.cuda.is_available() and args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        action_chunk, info = predict_action(
                            model=model,
                            processor=None,
                            image=Image.fromarray(primary).convert("RGB"),
                            wrist_image=Image.fromarray(wrist).convert("RGB"),
                            proprio=calvin_proprio(obs["robot_obs"]),
                            instruction=instruction,
                            unnorm_key=CALVIN_DATASET_NAME,
                            device=args.device,
                            update_history=True,
                            center_crop=True,
                        )
                        if torch.cuda.is_available() and args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        latencies_ms.append((time.perf_counter() - start_time) * 1000.0)
                        reasoning_steps.append(_reasoning_steps(info))
                        rows = np.asarray(action_chunk, dtype=np.float32).reshape(-1, 7)
                        queue.extend(rows[:NUM_ACTIONS_CHUNK])
                    obs, _, _, current_info = env.step(process_calvin_action(queue.popleft()))
                    if task_oracle.get_task_info_for_set(start_info, current_info, {task}):
                        success = True
                        break
                if not success:
                    break
                completed += 1
            results.append(completed)
    finally:
        env.close()

    payload = {
        "benchmark": "calvin_abc_to_d",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "num_sequences": args.num_sequences,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sequence_indices": selected_indices,
        "sequence_results": results,
        **chain_metrics(results),
        "policy_queries": len(latencies_ms),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)) if latencies_ms else 0.0,
        "avg_reasoning_steps": float(np.mean(reasoning_steps)) if reasoning_steps else 0.0,
        "exit_threshold": args.exit_threshold,
        "max_reasoning_steps": args.max_reasoning_steps,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--sequence-workers", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=360)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exit-threshold", type=float, default=0.55)
    parser.add_argument("--max-reasoning-steps", type=int, default=5)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid CALVIN shard specification")
    if not args.protocol_only and (args.checkpoint is None or args.dataset_root is None):
        raise ValueError("CALVIN rollout evaluation needs --checkpoint and --dataset-root")
    payload = evaluate(args)
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
