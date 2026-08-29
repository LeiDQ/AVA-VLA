"""Small dependency-free bridge to the official CALVIN ABC->D protocol."""

from __future__ import annotations

import contextlib
import sys
import types
from math import pi
from typing import Dict

import numpy as np

from experiments.robot.calvin.calvin_utils import add_official_calvin_to_path


@contextlib.contextmanager
def temp_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def official_sequences(num_sequences: int = 1000, num_workers: int = 1):
    """Call the pinned official generator while avoiding its training-only imports."""

    add_official_calvin_to_path()
    module_name = "calvin_agent.evaluation.utils"
    previous = sys.modules.get(module_name)
    bridge = types.ModuleType(module_name)
    bridge.temp_seed = temp_seed
    sys.modules[module_name] = bridge
    try:
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        return get_sequences(int(num_sequences), num_workers=int(num_workers))
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def _fnv1_32(text: str) -> int:
    # pyhash.fnv1_32(), used by the official evaluator, passes seed=0 rather
    # than the canonical FNV offset basis.  Preserve that exact behavior.
    value = 0
    for byte in text.encode("utf-8"):
        value = (value * 0x01000193) & 0xFFFFFFFF
        value ^= byte
    return value


def initial_condition_state(initial_condition: Dict):
    """Translate an official sequence initial condition into simulator state."""

    robot_obs = np.array(
        [
            0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596,
            1.50013585, 0.07999963, -1.21779124, 1.03987629, 2.11978254,
            -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0,
        ]
    )
    rotation_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    with temp_seed(_fnv1_32(str(initial_condition.values()))):
        np.random.shuffle(table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        positions = {"slider_right": slider_right, "slider_left": slider_left}
        scene_obs[6:9] = positions.get(initial_condition["red_block"], table[0])
        scene_obs[11] = np.random.uniform(*rotation_range)
        if initial_condition["blue_block"] in positions:
            scene_obs[12:15] = positions[initial_condition["blue_block"]]
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = table[1]
        else:
            scene_obs[12:15] = table[0]
        scene_obs[17] = np.random.uniform(*rotation_range)
        scene_obs[18:21] = positions.get(initial_condition["pink_block"], table[1])
        scene_obs[23] = np.random.uniform(*rotation_range)
    return robot_obs, scene_obs


def chain_metrics(results):
    values = np.asarray(results, dtype=np.int64)
    if values.size == 0:
        raise ValueError("No CALVIN sequence results supplied")
    return {
        "success_rate_1": float(np.mean(values >= 1)),
        "success_rate_2": float(np.mean(values >= 2)),
        "success_rate_3": float(np.mean(values >= 3)),
        "success_rate_4": float(np.mean(values >= 4)),
        "success_rate_5": float(np.mean(values >= 5)),
        "avg_sequence_length": float(np.mean(values)),
    }
