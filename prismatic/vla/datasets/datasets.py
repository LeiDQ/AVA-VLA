"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        actions = rlds_batch["action"]
        current_action_idx = max(0, actions.shape[0] - NUM_ACTIONS_CHUNK)
        current_action = actions[current_action_idx]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = actions[current_action_idx + 1 :]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        action_chunk = actions[current_action_idx : current_action_idx + NUM_ACTIONS_CHUNK]
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=action_chunk,
        )

        history_actions = actions[:current_action_idx]
        history_parts = [history_actions]
        if current_action_idx > 0 and "proprio" in rlds_batch["observation"]:
            history_proprio = rlds_batch["observation"]["proprio"][:current_action_idx]
            history_parts.append(history_proprio)
        if current_action_idx > 0:
            return_dict["history_states"] = np.concatenate(history_parts, axis=-1).astype(np.float32)
            return_dict["history_pad_mask"] = rlds_batch["observation"].get(
                "pad_mask", np.ones(current_action_idx, dtype=bool)
            )[:current_action_idx].astype(bool)

        for key in ("reward", "success"):
            if key in rlds_batch:
                signal = np.asarray(rlds_batch[key][current_action_idx : current_action_idx + NUM_ACTIONS_CHUNK])
                return_dict["task_rewards"] = np.asarray(np.max(signal), dtype=np.float32)
                break

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][-1])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][-1]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        window_size: int = 1,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # Past observations/actions for h_{t-1}
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_root_dir: Optional[Union[str, Path]] = None,
        metadata_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.data_root_dir = Path(data_root_dir) if data_root_dir is not None else None
        self.metadata_file = Path(metadata_file) if metadata_file is not None else None
        self.samples: List[Dict[str, Any]] = []

        if self.data_root_dir is not None:
            self._load_samples_from_disk()

        if self.samples:
            actions = np.stack([sample["action"] for sample in self.samples])
            self.dataset_statistics = {
                "dummy_dataset": {
                    "action": {
                        "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
                        "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
                    }
                }
            }
        else:
            self.dataset_statistics = {
                "dummy_dataset": {
                    "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
                }
            }

    def _load_samples_from_disk(self) -> None:
        if self.metadata_file is None:
            for candidate in ("metadata.jsonl", "metadata.json", "metadata.csv"):
                candidate_path = self.data_root_dir / candidate
                if candidate_path.exists():
                    self.metadata_file = candidate_path
                    break

        if self.metadata_file is None or not self.metadata_file.exists():
            return

        if self.metadata_file.suffix == ".jsonl":
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif self.metadata_file.suffix == ".json":
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "samples" in data and isinstance(data["samples"], list):
                rows = data["samples"]
            elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                rows = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            else:
                raise ValueError(f"Unsupported JSON metadata format in `{self.metadata_file}`")
        elif self.metadata_file.suffix == ".csv":
            with open(self.metadata_file, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError(f"Unsupported metadata extension: {self.metadata_file.suffix}")

        for row in rows:
            image_path = Path(row.get("image") or row.get("image_path") or row.get("image_filename"))
            if not image_path.is_absolute():
                image_path = self.data_root_dir / image_path

            action = self._normalize_action(row.get("action") or row.get("actions"))
            instruction = row.get("instruction") or row.get("text") or row.get("language_instruction")
            if action is None or instruction is None:
                continue

            self.samples.append(
                {"image_path": image_path, "action": action, "instruction": instruction}
            )

    def _normalize_action(self, action_value: Any) -> Optional[np.ndarray]:
        if action_value is None:
            return None

        if isinstance(action_value, np.ndarray):
            action = action_value.astype(np.float32)
        elif isinstance(action_value, (list, tuple)):
            action = np.asarray(action_value, dtype=np.float32)
        elif isinstance(action_value, str):
            try:
                action = np.asarray(json.loads(action_value), dtype=np.float32)
            except json.JSONDecodeError:
                action = np.asarray([float(x.strip()) for x in action_value.split(",") if x.strip()], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action type `{type(action_value)}`")

        return action.flatten()

    def __len__(self):
        if self.samples:
            return len(self.samples)
        return 10000

    def __getitem__(self, idx):
        if self.samples:
            sample = self.samples[idx]
            image = Image.open(sample["image_path"]).convert("RGB")
            action = sample["action"]
            instruction = sample["instruction"]
        else:
            image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
            action = np.asarray(np.random.rand(7), dtype=np.float32)
            instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
