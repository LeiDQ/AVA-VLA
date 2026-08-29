#!/usr/bin/env python3
"""Validate native AVA-VLA CALVIN parsing and the official simulator on debug data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for source in (
    ROOT,
    ROOT / "third_party/calvin_runtime",
    ROOT / "third_party/calvin/calvin_models",
    ROOT / "third_party/calvin/calvin_env",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from experiments.robot.calvin.calvin_utils import calvin_images, calvin_proprio, resolve_calvin_dataset_root
from experiments.robot.calvin.dataset import CalvinDiskDataset
from experiments.robot.calvin.online_rollout import CalvinVectorRollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = resolve_calvin_dataset_root(args.dataset_root)
    dataset = CalvinDiskDataset(
        root,
        "calvin_abc",
        batch_transform=lambda row: row,
        window_size=9,
        train=True,
        seed=0,
    )
    start, end, instruction, task = dataset.segments[0]
    sample = dataset.raw_sample(0, start)
    assert sample["action"].shape == (8, 7)
    assert sample["observation"]["proprio"].shape == (9, 8)
    assert sample["observation"]["image_primary"].ndim == 4
    assert sample["observation"]["image_wrist"].ndim == 4

    collector = CalvinVectorRollout(
        dataset_root=root,
        num_envs=1,
        rank=0,
        world_size=1,
        seed=0,
        max_steps=1,
    )
    try:
        observation = collector.observations()[0]
        assert observation.unnorm_key == "calvin_abc"
        assert observation.proprio.shape == (8,)
        primary_shape = list(observation.image.shape)
        wrist_shape = list(observation.wrist_image.shape)
        actions = np.zeros((1, 8, 7), dtype=np.float32)
        actions[..., -1] = 1.0
        rewards, terminals, lengths, env_steps, metrics = collector.step_action_chunks(
            actions,
            np.zeros((1, 16), dtype=np.float32),
        )
        assert rewards.shape == terminals.shape == lengths.shape == (1,)
        assert env_steps == 1 and terminals[0]
        assert metrics["online_completed_episodes"] == 1.0
    finally:
        collector.close()

    with np.load(root / "training" / f"episode_{start:07d}.npz") as frame:
        raw_primary, raw_wrist = calvin_images(
            {"rgb_obs": {"rgb_static": frame["rgb_static"], "rgb_gripper": frame["rgb_gripper"]}}
        )
        raw_proprio = calvin_proprio(frame["robot_obs"])
    payload = {
        "status": "PASS",
        "dataset_root": str(root),
        "training_frames_downloaded": len(list((root / "training").glob("episode_*.npz"))),
        "validation_frames_downloaded": len(list((root / "validation").glob("episode_*.npz"))),
        "first_segment": {"start": start, "end": end, "instruction": instruction, "task": task},
        "static_image_shape": list(raw_primary.shape),
        "gripper_image_shape": list(raw_wrist.shape),
        "proprio_shape": list(raw_proprio.shape),
        "action_chunk_shape": list(sample["action"].shape),
        "simulator_static_shape": primary_shape,
        "simulator_gripper_shape": wrist_shape,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
