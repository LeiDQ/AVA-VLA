"""Shared adapters for the official CALVIN observation and action formats."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


CALVIN_DATASET_NAME = "calvin_abc"
CALVIN_ACTION_DIM = 7
CALVIN_ROBOT_OBS_DIM = 15
CALVIN_PROPRIO_DIM = 8
CALVIN_CAMERA_OBS_SPACE = {
    "rgb_obs": ["rgb_static", "rgb_gripper"],
    "depth_obs": [],
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def add_official_calvin_to_path() -> Tuple[Path, Path]:
    """Expose the pinned official CALVIN and calvin_env source trees.

    Dependencies are deliberately not installed into the shared AVA-VLA virtual
    environment.  A launcher may prepend ``third_party/calvin_runtime`` for the
    small set of simulator dependencies installed inside this audit copy.
    """

    root = repository_root()
    model_source = root / "third_party" / "calvin" / "calvin_models"
    env_source = root / "third_party" / "calvin" / "calvin_env"
    runtime = root / "third_party" / "calvin_runtime"
    for path in (runtime, model_source, env_source):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return model_source, env_source


def resolve_calvin_dataset_root(path: Path) -> Path:
    """Resolve an extracted official dataset root without guessing silently."""

    path = Path(path).expanduser().resolve()
    candidates = [path, path / "calvin_debug_dataset", path / "task_ABC_D"]
    for candidate in candidates:
        if (candidate / "training").is_dir() and (candidate / "validation").is_dir():
            return candidate
    raise FileNotFoundError(
        f"No official CALVIN dataset found below {path}; expected training/ and validation/"
    )


def calvin_proprio(robot_obs: np.ndarray) -> np.ndarray:
    """Map CALVIN's 15-D robot state to AVA-VLA's explicit 8-D state.

    The representation is TCP position (3), TCP Euler orientation (3), gripper
    opening width (1), and the binary gripper command/state (1).  Joint angles
    are omitted so training and online inference have exactly the same contract.
    """

    values = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if values.size != CALVIN_ROBOT_OBS_DIM:
        raise ValueError(
            f"CALVIN robot_obs must have {CALVIN_ROBOT_OBS_DIM} values, got {values.size}"
        )
    return np.concatenate([values[:7], values[-1:]]).astype(np.float32)


def calvin_images(observation: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Read static and gripper RGB arrays from raw simulator observations."""

    rgb = observation.get("rgb_obs", observation)
    try:
        primary = np.asarray(rgb["rgb_static"], dtype=np.uint8)
        wrist = np.asarray(rgb["rgb_gripper"], dtype=np.uint8)
    except KeyError as error:
        raise KeyError("CALVIN observation is missing rgb_static or rgb_gripper") from error
    return primary, wrist


def process_calvin_action(action: np.ndarray) -> np.ndarray:
    """Clamp a predicted relative action and binarize CALVIN's gripper command."""

    values = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if values.size != CALVIN_ACTION_DIM:
        raise ValueError(f"CALVIN relative action must have 7 values, got {values.size}")
    values[:6] = np.clip(values[:6], -1.0, 1.0)
    values[-1] = 1.0 if values[-1] > 0.0 else -1.0
    return values
