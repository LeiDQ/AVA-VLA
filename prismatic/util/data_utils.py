"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def pad_sequence_with_side(
    tensors: Sequence[torch.Tensor], batch_first: bool, padding_value: int, padding_side: str
) -> torch.Tensor:
    if padding_side == "right":
        return pad_sequence(tensors, batch_first=batch_first, padding_value=padding_value)

    if padding_side == "left":
        max_len = max(t.size(0) for t in tensors)
        padded = tensors[0].new_full((len(tensors), max_len), padding_value)
        for idx, tensor in enumerate(tensors):
            padded[idx, max_len - tensor.size(0) :] = tensor
        return padded

    raise ValueError(f"Unsupported padding_side `{padding_side}`; expected 'left' or 'right'.")


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        input_ids = pad_sequence_with_side(
            input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side=self.padding_side
        )
        labels = pad_sequence_with_side(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side=self.padding_side)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        input_ids = pad_sequence_with_side(
            input_ids, batch_first=True, padding_value=self.pad_token_id, padding_side=self.padding_side
        )
        labels = pad_sequence_with_side(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side=self.padding_side)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values_wrist = None
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = torch.stack([instance["pixel_values_wrist"] for instance in instances])
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = {
                    k: torch.stack([instance["pixel_values_wrist"][k] for instance in instances])
                    for k in instances[0]["pixel_values_wrist"]
                }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = torch.as_tensor(np.stack([instance["proprio"] for instance in instances]), dtype=torch.float32)
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if pixel_values_wrist is not None:
            output["pixel_values_wrist"] = pixel_values_wrist
        if "pixel_values_history" in instances[0]:
            history_pixels = [instance["pixel_values_history"] for instance in instances]
            if isinstance(history_pixels[0], torch.Tensor):
                output["pixel_values_history"] = torch.stack(history_pixels)
            else:
                output["pixel_values_history"] = {
                    key: torch.stack([value[key] for value in history_pixels])
                    for key in history_pixels[0]
                }
            output["history_pad_mask"] = torch.as_tensor(
                [bool(instance["history_pad_mask"]) for instance in instances],
                dtype=torch.bool,
            )
            if "proprio_history" in instances[0]:
                output["proprio_history"] = torch.as_tensor(
                    np.stack([instance["proprio_history"] for instance in instances]),
                    dtype=torch.float32,
                )
            if "pixel_values_wrist_history" in instances[0]:
                wrist_history = [instance["pixel_values_wrist_history"] for instance in instances]
                if isinstance(wrist_history[0], torch.Tensor):
                    output["pixel_values_wrist_history"] = torch.stack(wrist_history)
                else:
                    output["pixel_values_wrist_history"] = {
                        key: torch.stack([value[key] for value in wrist_history])
                        for key in wrist_history[0]
                    }
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        if "history_states" in instances[0]:
            output["history_states"] = torch.stack(
                [torch.from_numpy(np.copy(instance["history_states"])) for instance in instances]
            )
            output["history_pad_mask"] = torch.stack(
                [
                    torch.from_numpy(np.copy(instance.get("history_pad_mask", np.ones(instance["history_states"].shape[0], dtype=bool))))
                    for instance in instances
                ]
            )
        if "task_rewards" in instances[0]:
            output["task_rewards"] = torch.as_tensor(
                [float(instance["task_rewards"]) for instance in instances], dtype=torch.float32
            )
        return output
