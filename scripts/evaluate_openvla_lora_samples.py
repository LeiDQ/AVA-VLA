#!/usr/bin/env python3
"""
Evaluate a LoRA fine-tuned OpenVLA checkpoint on a few LIBERO RLDS samples.

This script is intentionally small and offline-friendly: it reloads the base
OpenVLA model, applies the saved LoRA adapter plus continuous action/proprio
heads, runs real forward passes, and writes JSON metrics and predictions.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ModuleFacade(torch.nn.Module):
    """DDP-like facade used by OpenVLA helpers without starting distributed."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def strip_ddp_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def get_nested_attr(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, "model") and hasattr(obj.model, name):
        return getattr(obj.model, name)
    if hasattr(obj, "base_model") and hasattr(obj.base_model, "model") and hasattr(obj.base_model.model, name):
        return getattr(obj.base_model.model, name)
    raise AttributeError(f"Could not find attribute {name!r} on model")


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().float().cpu().tolist()


def load_base_model_path(vla_path: str) -> str:
    if model_is_on_hf_hub(vla_path):
        return snapshot_download(repo_id=vla_path)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    return vla_path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_model_path = load_base_model_path(args.vla_path)
    update_auto_map(base_model_path)
    check_model_logic_mismatch(base_model_path)

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    base_vla.vision_backbone.set_num_images_in_input(args.num_images_in_input)
    vla = PeftModel.from_pretrained(base_vla, checkpoint_dir / "lora_adapter").to(device)
    vla.eval()

    llm_dim = get_nested_attr(vla, "llm_dim")
    image_sizes = tuple(get_nested_attr(vla, "config").image_sizes)

    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    action_head.load_state_dict(
        strip_ddp_prefix(torch.load(checkpoint_dir / "action_head--latest_checkpoint.pt", map_location="cpu"))
    )
    action_head = action_head.to(torch.bfloat16).to(device).eval()

    proprio_projector = None
    if args.use_proprio:
        proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM)
        proprio_projector.load_state_dict(
            strip_ddp_prefix(torch.load(checkpoint_dir / "proprio_projector--latest_checkpoint.pt", map_location="cpu"))
        )
        proprio_projector = ModuleFacade(proprio_projector.to(device).eval())

    num_patches = get_nested_attr(vla, "vision_backbone").get_num_patches() * args.num_images_in_input
    if args.use_proprio:
        num_patches += 1

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=args.num_images_in_input > 1,
        use_proprio=args.use_proprio,
    )
    dataset = RLDSDataset(
        Path(args.data_root_dir),
        args.dataset_name,
        batch_transform,
        resize_resolution=image_sizes,
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(dataset, batch_size=1, sampler=None, collate_fn=collator, num_workers=0)

    samples = []
    aggregate = {
        "loss_value": [],
        "curr_action_l1_loss": [],
        "next_actions_l1_loss": [],
        "mean_action_l2": [],
    }

    with torch.no_grad():
        for sample_idx, batch in enumerate(dataloader):
            if sample_idx >= args.num_samples:
                break

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"].to(device),
                    output_hidden_states=True,
                    proprio=batch["proprio"].to(device) if args.use_proprio else None,
                    proprio_projector=proprio_projector if args.use_proprio else None,
                    use_film=False,
                )

                labels = batch["labels"][:, 1:].to(device)
                current_action_mask = get_current_action_mask(labels)
                next_actions_mask = get_next_actions_mask(labels)
                text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
                action_hidden_states = (
                    text_hidden_states[current_action_mask | next_actions_mask]
                    .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                    .to(torch.bfloat16)
                )
                predicted_actions = action_head.predict_action(action_hidden_states)

            ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)
            diff = predicted_actions.float() - ground_truth_actions.float()
            metrics = {
                "loss_value": F.l1_loss(predicted_actions.float(), ground_truth_actions.float()).item(),
                "curr_action_l1_loss": F.l1_loss(predicted_actions[:, 0].float(), ground_truth_actions[:, 0].float()).item(),
                "next_actions_l1_loss": F.l1_loss(
                    predicted_actions[:, 1:].float(), ground_truth_actions[:, 1:].float()
                ).item(),
                "mean_action_l2": torch.linalg.vector_norm(diff, dim=-1).mean().item(),
            }
            for key, value in metrics.items():
                aggregate[key].append(value)

            samples.append(
                {
                    "sample_index": sample_idx,
                    "instruction": batch.get("instruction", [""])[0] if isinstance(batch.get("instruction"), list) else "",
                    "metrics": metrics,
                    "predicted_actions": tensor_to_list(predicted_actions[0]),
                    "target_actions": tensor_to_list(ground_truth_actions[0]),
                }
            )

    summary = {f"mean_{key}": sum(values) / len(values) for key, values in aggregate.items() if values}
    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "base_model_path": base_model_path,
        "dataset_name": args.dataset_name,
        "data_root_dir": str(Path(args.data_root_dir).expanduser()),
        "num_samples": len(samples),
        "num_images_in_input": args.num_images_in_input,
        "use_proprio": args.use_proprio,
        "summary": summary,
        "samples": samples,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "summary": summary}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--vla_path", default="openvla/openvla-7b")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--dataset_name", default="libero_spatial_no_noops")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
