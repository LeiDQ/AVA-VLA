"""True online CALVIN rollouts for AVA-VLA Stage-3 PPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.robot.calvin.calvin_utils import (
    CALVIN_DATASET_NAME,
    CALVIN_CAMERA_OBS_SPACE,
    add_official_calvin_to_path,
    calvin_images,
    calvin_proprio,
    process_calvin_action,
    resolve_calvin_dataset_root,
)
from experiments.robot.calvin.dataset import _load_language_annotations


CALVIN_EP_LEN = 360


@dataclass
class OnlineObservation:
    image: np.ndarray
    wrist_image: np.ndarray
    proprio: np.ndarray
    instruction: str
    history_state: Optional[np.ndarray]
    unnorm_key: str
    suite_name: str


class CalvinVectorRollout:
    """Maintain independent official CALVIN simulators on one DDP rank."""

    def __init__(
        self,
        dataset_root: Path,
        num_envs: int,
        rank: int,
        world_size: int,
        seed: int,
        split: str = "training",
        max_steps: int = CALVIN_EP_LEN,
        **_: object,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("CALVIN online PPO requires at least one environment")
        if split not in {"training", "validation"}:
            raise ValueError(f"Unsupported CALVIN split: {split}")
        add_official_calvin_to_path()
        try:
            import hydra
            from omegaconf import OmegaConf
            from calvin_env.envs.play_table_env import get_env
        except ImportError as error:
            raise RuntimeError(
                "CALVIN simulator dependencies are missing. Run scripts/install_calvin_runtime.sh."
            ) from error

        self.root = resolve_calvin_dataset_root(Path(dataset_root))
        self.split_dir = self.root / split
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.segments = _load_language_annotations(self.split_dir)
        if not any(task for _, _, _, task in self.segments):
            raise ValueError("CALVIN online PPO needs task labels in auto_lang_ann.npy")

        config_root = Path(__file__).resolve().parents[3] / "third_party" / "calvin" / "calvin_models" / "conf"
        task_config = OmegaConf.load(config_root / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(task_config)
        self.slots: List[Dict] = []
        for local_index in range(self.num_envs):
            global_index = self.rank * self.num_envs + local_index
            env = get_env(
                self.split_dir,
                obs_space=CALVIN_CAMERA_OBS_SPACE,
                show_gui=False,
            )
            if hasattr(env, "seed"):
                env.seed(self.seed + global_index)
            slot = {
                "env": env,
                "global_index": global_index,
                "episode_index": global_index % len(self.segments),
                "episode_step": 0,
                "history_state": None,
                "obs": None,
                "instruction": "",
                "task": "",
                "start_info": None,
            }
            self._reset_slot(slot)
            self.slots.append(slot)

    def _load_frame(self, index: int) -> Dict[str, np.ndarray]:
        path = self.split_dir / f"episode_{int(index):07d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"CALVIN rollout reset frame is missing: {path}")
        with np.load(path) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}

    def _reset_slot(self, slot: Dict) -> None:
        segment_index = int(slot["episode_index"]) % len(self.segments)
        start, _, instruction, task = self.segments[segment_index]
        if not task:
            raise ValueError(f"CALVIN segment {segment_index} has no task label")
        frame = self._load_frame(start)
        slot["obs"] = slot["env"].reset(
            robot_obs=np.asarray(frame["robot_obs"]),
            scene_obs=np.asarray(frame["scene_obs"]),
        )
        slot["instruction"] = instruction
        slot["task"] = task
        slot["episode_step"] = 0
        slot["history_state"] = None
        slot["start_info"] = slot["env"].get_info()

    def observations(self) -> List[OnlineObservation]:
        rows = []
        for slot in self.slots:
            primary, wrist = calvin_images(slot["obs"])
            rows.append(
                OnlineObservation(
                    image=primary,
                    wrist_image=wrist,
                    proprio=calvin_proprio(slot["obs"]["robot_obs"]),
                    instruction=slot["instruction"],
                    history_state=slot["history_state"],
                    unnorm_key=CALVIN_DATASET_NAME,
                    suite_name="calvin_abc",
                )
            )
        return rows

    def step_action_chunks(
        self,
        action_chunks: np.ndarray,
        final_latents: np.ndarray,
    ):
        if len(action_chunks) != len(self.slots):
            raise ValueError("Action batch does not match CALVIN environment count")
        rewards = np.zeros(len(self.slots), dtype=np.float32)
        terminals = np.zeros(len(self.slots), dtype=bool)
        chunk_lengths = np.zeros(len(self.slots), dtype=np.int64)
        env_steps = 0
        episodes = 0
        successes = 0
        for env_index, slot in enumerate(self.slots):
            succeeded = False
            for action in action_chunks[env_index]:
                processed = process_calvin_action(action)
                obs, _, _, current_info = slot["env"].step(processed)
                slot["obs"] = obs
                slot["episode_step"] += 1
                chunk_lengths[env_index] += 1
                env_steps += 1
                task_info = self.task_oracle.get_task_info_for_set(
                    slot["start_info"], current_info, {slot["task"]}
                )
                if task_info:
                    rewards[env_index] = 1.0
                    succeeded = True
                    break
                if slot["episode_step"] >= self.max_steps:
                    break
            terminal = succeeded or slot["episode_step"] >= self.max_steps
            terminals[env_index] = terminal
            if terminal:
                episodes += 1
                successes += int(succeeded)
                slot["episode_index"] += self.world_size * self.num_envs
                self._reset_slot(slot)
            else:
                slot["history_state"] = np.asarray(final_latents[env_index], dtype=np.float32)
        metrics = {
            "online_completed_episodes": float(episodes),
            "online_completed_successes": float(successes),
            "online_success_rate": float(successes) / float(episodes) if episodes else 0.0,
            "online_calvin_abc_completed_episodes": float(episodes),
            "online_calvin_abc_completed_successes": float(successes),
            "online_calvin_abc_success_rate": float(successes) / float(episodes) if episodes else 0.0,
        }
        return rewards, terminals, chunk_lengths, env_steps, metrics

    def close(self) -> None:
        for slot in self.slots:
            try:
                slot["env"].close()
            except Exception:
                pass
