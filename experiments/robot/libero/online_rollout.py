"""Distributed online LIBERO rollout state used by AVA-VLA PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

SUITE_DATASET_KEYS = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
ALL_FOUR_SUITES_DATASET = "libero_4_task_suites_no_noops"


@dataclass
class OnlineObservation:
    image: np.ndarray
    wrist_image: np.ndarray
    proprio: np.ndarray
    instruction: str
    history_state: Optional[np.ndarray]
    unnorm_key: str
    suite_name: str


class LiberoVectorRollout:
    """Keep several independent LIBERO environments active on each DDP rank."""

    def __init__(
        self,
        task_suite_name: Union[str, Sequence[str]],
        num_envs: int,
        rank: int,
        world_size: int,
        seed: int,
        num_steps_wait: int = 10,
        env_img_res: int = 256,
    ) -> None:
        from libero.libero import benchmark

        task_suite_names = (
            (task_suite_name,)
            if isinstance(task_suite_name, str)
            else tuple(task_suite_name)
        )
        if not task_suite_names:
            raise ValueError("At least one online LIBERO suite is required")
        if len(set(task_suite_names)) != len(task_suite_names):
            raise ValueError(f"Duplicate online LIBERO suites: {task_suite_names}")
        unsupported = [name for name in task_suite_names if name not in TASK_MAX_STEPS]
        if unsupported:
            raise ValueError(f"Unsupported online LIBERO suites: {unsupported}")
        if int(num_envs) < len(task_suite_names):
            raise ValueError(
                "num_envs must be at least the number of online LIBERO suites so every suite "
                "is represented on every DDP rank"
            )
        if int(num_envs) % len(task_suite_names):
            raise ValueError(
                "Equal-weight multi-suite PPO requires num_envs to be divisible by the number of suites"
            )

        self.task_suite_names: Tuple[str, ...] = task_suite_names
        self.task_suite_name = (
            task_suite_names[0]
            if len(task_suite_names) == 1
            else "libero_4_task_suites"
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.num_steps_wait = int(num_steps_wait)
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suites = {
            name: benchmark_dict[name]() for name in self.task_suite_names
        }
        self.task_suite = self.task_suites[self.task_suite_names[0]]
        self.max_steps = max(TASK_MAX_STEPS[name] for name in self.task_suite_names)
        self.slots: List[Dict] = []

        for local_index in range(int(num_envs)):
            global_index = self.rank * int(num_envs) + local_index
            suite_index = global_index % len(self.task_suite_names)
            suite_name = self.task_suite_names[suite_index]
            task_suite = self.task_suites[suite_name]
            suite_global_index = global_index // len(self.task_suite_names)
            task_id = suite_global_index % task_suite.n_tasks
            task = task_suite.get_task(task_id)
            env, instruction = get_libero_env(task, "avavla", resolution=env_img_res)
            env.seed(self.seed + global_index)
            slot = {
                "env": env,
                "suite_name": suite_name,
                "unnorm_key": SUITE_DATASET_KEYS[suite_name],
                "max_steps": TASK_MAX_STEPS[suite_name],
                "task_id": task_id,
                "instruction": instruction,
                "initial_states": task_suite.get_task_init_states(task_id),
                "episode_index": (suite_global_index // task_suite.n_tasks) % 50,
                "episode_step": 0,
                "history_state": None,
                "obs": None,
                "successes": 0,
                "episodes": 0,
            }
            self._reset_slot(slot)
            self.slots.append(slot)

    @staticmethod
    def suite_for_dataset(dataset_name: str) -> str:
        suites = LiberoVectorRollout.suites_for_dataset(dataset_name)
        if len(suites) != 1:
            raise ValueError(
                f"dataset_name={dataset_name!r} maps to multiple LIBERO suites; "
                "use suites_for_dataset()"
            )
        return suites[0]

    @staticmethod
    def suites_for_dataset(dataset_name: str) -> Tuple[str, ...]:
        if dataset_name == ALL_FOUR_SUITES_DATASET:
            return tuple(TASK_MAX_STEPS)
        for suite in TASK_MAX_STEPS:
            if dataset_name == suite or dataset_name.startswith(f"{suite}_"):
                return (suite,)
        raise ValueError(
            f"Cannot infer LIBERO task suite from dataset_name={dataset_name!r}; "
            f"expected one of {tuple(TASK_MAX_STEPS)} or {ALL_FOUR_SUITES_DATASET!r}."
        )

    def _reset_slot(self, slot: Dict) -> None:
        env = slot["env"]
        env.reset()
        initial_states = slot["initial_states"]
        state_index = int(slot["episode_index"]) % len(initial_states)
        obs = env.set_init_state(initial_states[state_index])
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("avavla"))
        slot["obs"] = obs
        slot["episode_step"] = 0
        slot["history_state"] = None

    def observations(self) -> List[OnlineObservation]:
        rows: List[OnlineObservation] = []
        for slot in self.slots:
            obs = slot["obs"]
            proprio = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ).astype(np.float32)
            rows.append(
                OnlineObservation(
                    image=get_libero_image(obs),
                    wrist_image=get_libero_wrist_image(obs),
                    proprio=proprio,
                    instruction=slot["instruction"],
                    history_state=slot["history_state"],
                    unnorm_key=slot["unnorm_key"],
                    suite_name=slot["suite_name"],
                )
            )
        return rows

    @staticmethod
    def _process_action(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(np.asarray(action).copy(), binarize=True)
        return invert_gripper_action(action)

    def step_action_chunks(
        self,
        action_chunks: np.ndarray,
        final_latents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict[str, float]]:
        """Execute one action chunk and expose its exact semi-Markov duration."""
        if action_chunks.shape[0] != len(self.slots):
            raise ValueError("Action batch does not match the number of online LIBERO environments")

        rewards = np.zeros(len(self.slots), dtype=np.float32)
        terminals = np.zeros(len(self.slots), dtype=bool)
        chunk_lengths = np.zeros(len(self.slots), dtype=np.int64)
        env_steps = 0
        completed_episodes = 0
        completed_successes = 0
        suite_completed_episodes = {name: 0 for name in self.task_suite_names}
        suite_completed_successes = {name: 0 for name in self.task_suite_names}
        for env_index, slot in enumerate(self.slots):
            succeeded = False
            for action in action_chunks[env_index]:
                processed = self._process_action(action)
                obs, reward, done, _ = slot["env"].step(processed.tolist())
                slot["obs"] = obs
                slot["episode_step"] += 1
                env_steps += 1
                chunk_lengths[env_index] += 1
                if bool(done) or float(reward) > 0.0:
                    succeeded = True
                    rewards[env_index] = 1.0
                    break
                if slot["episode_step"] >= slot["max_steps"]:
                    break

            terminal = succeeded or slot["episode_step"] >= slot["max_steps"]
            terminals[env_index] = terminal
            if terminal:
                slot["episodes"] += 1
                slot["successes"] += int(succeeded)
                completed_episodes += 1
                completed_successes += int(succeeded)
                suite_name = slot["suite_name"]
                suite_completed_episodes[suite_name] += 1
                suite_completed_successes[suite_name] += int(succeeded)
                slot["episode_index"] += self.world_size
                self._reset_slot(slot)
            else:
                slot["history_state"] = np.asarray(final_latents[env_index], dtype=np.float32)

        metrics = {
            "online_completed_episodes": float(completed_episodes),
            "online_completed_successes": float(completed_successes),
            "online_success_rate": (
                float(completed_successes) / float(completed_episodes) if completed_episodes else 0.0
            ),
        }
        for suite_name in self.task_suite_names:
            episodes = suite_completed_episodes[suite_name]
            successes = suite_completed_successes[suite_name]
            metrics[f"online_{suite_name}_completed_episodes"] = float(episodes)
            metrics[f"online_{suite_name}_completed_successes"] = float(successes)
            metrics[f"online_{suite_name}_success_rate"] = (
                float(successes) / float(episodes) if episodes else 0.0
            )
        return rewards, terminals, chunk_lengths, env_steps, metrics

    def close(self) -> None:
        for slot in self.slots:
            try:
                slot["env"].close()
            except Exception:
                pass
