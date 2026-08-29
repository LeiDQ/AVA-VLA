"""Streaming PyTorch adapter for the official frame-wise CALVIN dataset."""

from __future__ import annotations

import fcntl
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from experiments.robot.calvin.calvin_utils import (
    CALVIN_DATASET_NAME,
    calvin_proprio,
    resolve_calvin_dataset_root,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _summary(values: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    return {
        "mean": values.mean(axis=0).astype(np.float32),
        "std": values.std(axis=0).astype(np.float32),
        "min": values.min(axis=0).astype(np.float32),
        "max": values.max(axis=0).astype(np.float32),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32),
        "mask": mask.astype(bool),
    }


def _load_language_annotations(split_dir: Path) -> List[Tuple[int, int, str, str]]:
    annotation_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing CALVIN language annotations: {annotation_path}")
    payload = np.load(annotation_path, allow_pickle=True).item()
    intervals = np.asarray(payload["info"]["indx"], dtype=np.int64)
    annotations = payload["language"]
    text_rows = list(annotations["ann"])
    task_rows = list(annotations.get("task", [""] * len(text_rows)))
    if len(intervals) != len(text_rows) or len(task_rows) != len(text_rows):
        raise ValueError("CALVIN annotation intervals, language, and task labels disagree")
    return [
        (int(bounds[0]), int(bounds[1]), str(text), str(task))
        for bounds, text, task in zip(intervals, text_rows, task_rows)
    ]


class CalvinDiskDataset(IterableDataset):
    """Emit OpenVLA/AVA-VLA samples directly from official CALVIN ``.npz`` frames.

    No converted copy is produced.  Samples contain nine observation frames
    (the current frame plus the previous AVA decision history) and eight future
    relative actions.  Boundary frames are repeated, matching RLDS padding.
    """

    STATS_FILE = "avavla_calvin_statistics.json"

    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str,
        batch_transform,
        window_size: int,
        train: bool = True,
        seed: int = 0,
        frame_cache_size: int = 256,
    ) -> None:
        if dataset_name != CALVIN_DATASET_NAME:
            raise ValueError(
                f"Native CALVIN adapter expects dataset_name={CALVIN_DATASET_NAME!r}, got {dataset_name!r}"
            )
        if int(window_size) < NUM_ACTIONS_CHUNK + 1:
            raise ValueError(
                f"CALVIN history window must be at least {NUM_ACTIONS_CHUNK + 1} frames"
            )
        self.root = resolve_calvin_dataset_root(Path(data_root_dir))
        self.split = "training" if train else "validation"
        self.split_dir = self.root / self.split
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.window_size = int(window_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.frame_cache_size = max(1, int(frame_cache_size))
        self._frame_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        self.segments = _load_language_annotations(self.split_dir)
        if not self.segments:
            raise ValueError(f"CALVIN split has no language segments: {self.split_dir}")
        self.dataset_length = int(sum(end - start + 1 for start, end, _, _ in self.segments))
        self.dataset_statistics = self.load_or_compute_statistics(self.root, dataset_name)

    @classmethod
    def load_or_compute_statistics(cls, root: Path, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        root = resolve_calvin_dataset_root(root)
        stats_path = root / cls.STATS_FILE
        lock_path = root / f".{cls.STATS_FILE}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if stats_path.is_file():
                with stats_path.open("r", encoding="utf-8") as stream:
                    return json.load(stream)

            frame_paths = sorted((root / "training").glob("episode_*.npz"))
            if not frame_paths:
                raise FileNotFoundError(f"No CALVIN episode frames in {root / 'training'}")
            actions = np.empty((len(frame_paths), 7), dtype=np.float32)
            proprio = np.empty((len(frame_paths), 8), dtype=np.float32)
            for index, frame_path in enumerate(frame_paths):
                with np.load(frame_path) as frame:
                    actions[index] = np.asarray(frame["rel_actions"], dtype=np.float32)
                    proprio[index] = calvin_proprio(frame["robot_obs"])
            stats = {
                dataset_name: {
                    "action": _summary(
                        actions,
                        np.asarray([True, True, True, True, True, True, False]),
                    ),
                    "proprio": _summary(proprio, np.ones(8, dtype=bool)),
                    "num_transitions": int(len(frame_paths)),
                    "num_trajectories": int(len(_load_language_annotations(root / "training"))),
                }
            }
            temporary = stats_path.with_name(f".{stats_path.name}.tmp-{os.getpid()}")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(_jsonable(stats), stream, indent=2)
                stream.write("\n")
            os.replace(temporary, stats_path)
            return _jsonable(stats)

    def _load_frame(self, index: int) -> Dict[str, np.ndarray]:
        cached = self._frame_cache.pop(int(index), None)
        if cached is not None:
            self._frame_cache[int(index)] = cached
            return cached
        frame_path = self.split_dir / f"episode_{int(index):07d}.npz"
        if not frame_path.is_file():
            raise FileNotFoundError(f"CALVIN annotation references missing frame {frame_path}")
        with np.load(frame_path) as payload:
            frame = {key: np.asarray(payload[key]) for key in payload.files}
        self._frame_cache[int(index)] = frame
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def _normalize(self, values: np.ndarray, field: str) -> np.ndarray:
        stats = self.dataset_statistics[self.dataset_name][field]
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
        scaled = 2.0 * (values - low) / (high - low + 1e-8) - 1.0
        return np.where(mask, np.clip(scaled, -1.0, 1.0), values).astype(np.float32)

    def raw_sample(self, segment_index: int, current_index: Optional[int] = None) -> Dict[str, Any]:
        start, end, instruction, _ = self.segments[int(segment_index)]
        if current_index is None:
            current_index = start
        current_index = int(current_index)
        if not start <= current_index <= end:
            raise IndexError(f"Frame {current_index} is outside language segment [{start}, {end}]")

        raw_history_indices = [current_index - offset for offset in reversed(range(self.window_size))]
        history_indices = [max(start, index) for index in raw_history_indices]
        action_indices = [min(end, current_index + offset) for offset in range(NUM_ACTIONS_CHUNK)]
        history = [self._load_frame(index) for index in history_indices]
        future = [self._load_frame(index) for index in action_indices]
        actions = self._normalize(
            np.stack([np.asarray(frame["rel_actions"], dtype=np.float32) for frame in future]),
            "action",
        )
        proprio = self._normalize(
            np.stack([calvin_proprio(frame["robot_obs"]) for frame in history]),
            "proprio",
        )
        return {
            "dataset_name": self.dataset_name,
            "action": actions,
            "observation": {
                "image_primary": np.stack([frame["rgb_static"] for frame in history]),
                "image_wrist": np.stack([frame["rgb_gripper"] for frame in history]),
                "proprio": proprio,
                "pad_mask": np.asarray([index >= start for index in raw_history_indices], dtype=bool),
            },
            "task": {"language_instruction": instruction.encode("utf-8")},
        }

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        worker = get_worker_info()
        workers_per_rank = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        shard_count = world_size * workers_per_rank
        shard_index = rank * workers_per_rank + worker_id
        assigned = np.arange(len(self.segments), dtype=np.int64)[shard_index::shard_count]
        if assigned.size == 0:
            raise RuntimeError(
                f"CALVIN has {len(self.segments)} language segments but {shard_count} data shards"
            )
        rng = np.random.default_rng(self.seed + shard_index)
        while True:
            order = assigned.copy()
            if self.train:
                rng.shuffle(order)
            for segment_index in order:
                start, end, _, _ = self.segments[int(segment_index)]
                current = int(rng.integers(start, end + 1)) if self.train else start
                yield self.batch_transform(self.raw_sample(int(segment_index), current))
            if not self.train:
                break

    def __len__(self) -> int:
        return self.dataset_length
