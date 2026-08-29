"""Regression tests for the Table-1 one-policy-for-all-four-suites rollout contract."""

from __future__ import annotations

import sys
import types
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "LIBERO"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".libero"))

from experiments.robot.libero import online_rollout as rollout_module  # noqa: E402
from vla_scripts.online_policy import (  # noqa: E402
    _normalize_proprio_batch,
    _resolve_unnorm_keys,
    _unnormalize_action_batch,
)


class _FakeTaskSuite:
    n_tasks = 10

    def get_task(self, task_id):
        return SimpleNamespace(task_id=task_id)

    def get_task_init_states(self, task_id):
        return np.arange(50, dtype=np.float32)[:, None] + task_id


class _FakeEnv:
    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return None

    def set_init_state(self, state):
        return {"state": np.asarray(state)}

    def step(self, action):
        del action
        return {"state": np.zeros(1)}, 0.0, False, {}

    def close(self):
        return None


class _FakeModel:
    def normalize_proprio(self, proprio, key):
        offset = {"suite_a": 1.0, "suite_b": 10.0}[key]
        return np.asarray(proprio) + offset

    def get_action_stats(self, key):
        if key == "suite_a":
            return {
                "q01": np.zeros(7, dtype=np.float32),
                "q99": np.full(7, 2.0, dtype=np.float32),
            }
        return {
            "q01": np.full(7, -10.0, dtype=np.float32),
            "q99": np.full(7, 10.0, dtype=np.float32),
        }


def test_dataset_mapping() -> None:
    suites = rollout_module.LiberoVectorRollout.suites_for_dataset(
        rollout_module.ALL_FOUR_SUITES_DATASET
    )
    assert suites == tuple(rollout_module.TASK_MAX_STEPS)
    assert rollout_module.LiberoVectorRollout.suite_for_dataset(
        "libero_spatial_no_noops"
    ) == "libero_spatial"
    try:
        rollout_module.LiberoVectorRollout.suite_for_dataset(
            rollout_module.ALL_FOUR_SUITES_DATASET
        )
    except ValueError as error:
        assert "multiple LIBERO suites" in str(error)
    else:
        raise AssertionError("The single-suite resolver accepted the four-suite mixture")


def test_equal_suite_allocation() -> None:
    benchmark_module = SimpleNamespace(
        get_benchmark_dict=lambda: {
            name: _FakeTaskSuite for name in rollout_module.TASK_MAX_STEPS
        }
    )
    libero_package = types.ModuleType("libero")
    libero_submodule = types.ModuleType("libero.libero")
    libero_submodule.benchmark = benchmark_module
    libero_package.libero = libero_submodule
    previous_modules = {
        name: sys.modules.get(name) for name in ("libero", "libero.libero")
    }
    original_get_env = rollout_module.get_libero_env
    original_dummy_action = rollout_module.get_libero_dummy_action
    try:
        sys.modules["libero"] = libero_package
        sys.modules["libero.libero"] = libero_submodule
        rollout_module.get_libero_env = lambda task, family, resolution: (
            _FakeEnv(),
            f"task-{task.task_id}",
        )
        rollout_module.get_libero_dummy_action = lambda family: np.zeros(7)
        collector = rollout_module.LiberoVectorRollout(
            task_suite_name=tuple(rollout_module.TASK_MAX_STEPS),
            num_envs=8,
            rank=0,
            world_size=8,
            seed=7,
            num_steps_wait=0,
        )
    finally:
        rollout_module.get_libero_env = original_get_env
        rollout_module.get_libero_dummy_action = original_dummy_action
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    suite_names = [slot["suite_name"] for slot in collector.slots]
    assert suite_names == list(rollout_module.TASK_MAX_STEPS) * 2
    for suite_name in rollout_module.TASK_MAX_STEPS:
        assert suite_names.count(suite_name) == 2
    assert [slot["unnorm_key"] for slot in collector.slots] == [
        rollout_module.SUITE_DATASET_KEYS[name] for name in suite_names
    ]
    for slot in collector.slots:
        slot["max_steps"] = 1
    _, _, _, env_steps, metrics = collector.step_action_chunks(
        np.zeros((8, 8, 7), dtype=np.float32),
        np.zeros((8, 4), dtype=np.float32),
    )
    assert env_steps == 8
    assert metrics["online_completed_episodes"] == 8.0
    for suite_name in rollout_module.TASK_MAX_STEPS:
        assert metrics[f"online_{suite_name}_completed_episodes"] == 2.0
        assert metrics[f"online_{suite_name}_completed_successes"] == 0.0


def test_per_observation_normalization() -> None:
    observations = [
        SimpleNamespace(proprio=np.zeros(8, dtype=np.float32), unnorm_key="suite_a"),
        SimpleNamespace(proprio=np.zeros(8, dtype=np.float32), unnorm_key="suite_b"),
    ]
    keys = _resolve_unnorm_keys(observations, None)
    assert keys == ["suite_a", "suite_b"]
    normalized = _normalize_proprio_batch(_FakeModel(), observations, keys)
    assert np.allclose(normalized[0], 1.0)
    assert np.allclose(normalized[1], 10.0)

    normalized_actions = np.zeros((2, 8, 7), dtype=np.float32)
    actions = _unnormalize_action_batch(_FakeModel(), normalized_actions, keys)
    assert np.allclose(actions[0], 1.0)
    assert np.allclose(actions[1], 0.0)


def main() -> int:
    test_dataset_mapping()
    test_equal_suite_allocation()
    test_per_observation_normalization()
    print("PASS: multi-suite allocation and per-observation normalization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
