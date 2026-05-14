"""
deploy_avavla.py

Deploy AVA-VLA for inference with latent reasoning and early-exit.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.conf import ModelConfig
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlas.avavla import AVAVLA
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)


def _checkpoint_sort_key(path: Path) -> tuple[int, object]:
    """Sort component checkpoints by numeric step when present, otherwise by mtime/name."""
    suffix = path.name.split("--", 1)[-1].split("_checkpoint.pt", 1)[0]
    if suffix.isdigit():
        return (0, int(suffix))
    try:
        return (1, path.stat().st_mtime)
    except OSError:
        return (1, path.name)


def _find_component_checkpoint(checkpoint_path: Path, stem: str) -> Optional[Path]:
    """Find latest or step-specific component checkpoint in a checkpoint directory."""
    latest = checkpoint_path / f"{stem}--latest_checkpoint.pt"
    if latest.exists():
        return latest

    candidates = sorted(checkpoint_path.glob(f"{stem}--*_checkpoint.pt"), key=_checkpoint_sort_key)
    return candidates[-1] if candidates else None


def _load_avavla_config(checkpoint_path: Path) -> Dict:
    """Load AVA-VLA hyperparameters saved by finetune_avavla.py."""
    config_path = checkpoint_path / "avavla_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def _unnormalize_actions(model: AVAVLA, normalized_actions: np.ndarray, unnorm_key: Optional[str]) -> np.ndarray:
    """Unnormalize continuous actions using dataset statistics."""
    action_norm_stats = model.get_action_stats(unnorm_key)
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS and "min" in action_norm_stats:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    else:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )


def load_avavla_model(
    checkpoint_path: Path,
    device: str = "cuda",
    enable_latent_reasoning: bool = True,
    max_reasoning_steps: Optional[int] = None,
    exit_threshold: Optional[float] = None,
) -> tuple:
    """
    Load AVA-VLA model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
        enable_latent_reasoning: Enable latent reasoning mechanism
        max_reasoning_steps: Maximum number of reasoning steps
        exit_threshold: Exit threshold for early-exit
    
    Returns:
        model: AVA-VLA model
        processor: Processor for inputs
        action_tokenizer: Action tokenizer
        norm_stats: Normalization statistics
    """
    checkpoint_path = Path(checkpoint_path)
    avavla_cfg = _load_avavla_config(checkpoint_path)
    max_reasoning_steps = max_reasoning_steps if max_reasoning_steps is not None else avavla_cfg.get("max_reasoning_steps", 5)
    exit_threshold = exit_threshold if exit_threshold is not None else avavla_cfg.get("exit_threshold", 0.8)

    # Load normalization statistics
    norm_stats_path = checkpoint_path / "dataset_statistics.json"
    if norm_stats_path.exists():
        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)
    else:
        norm_stats = {}
        print(f"Warning: No normalization statistics found at {norm_stats_path}")
    
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in AVA-VLA checkpoint directory: {checkpoint_path}")

    with open(config_path, "r") as f:
        config_json = json.load(f)

    if "vla" in config_json:
        model_cfg = ModelConfig.get_choice_class(config_json["vla"]["base_vlm"])()
        model_id = model_cfg.model_id
        vision_backbone_id = model_cfg.vision_backbone_id
        image_resize_strategy = model_cfg.image_resize_strategy
        llm_backbone_id = model_cfg.llm_backbone_id
        llm_max_length = model_cfg.llm_max_length
        arch_specifier = model_cfg.arch_specifier
    elif "model" in config_json:
        model_cfg = config_json["model"]
        model_id = model_cfg["model_id"]
        vision_backbone_id = model_cfg["vision_backbone_id"]
        image_resize_strategy = model_cfg["image_resize_strategy"]
        llm_backbone_id = model_cfg["llm_backbone_id"]
        llm_max_length = model_cfg.get("llm_max_length", 2048)
        arch_specifier = model_cfg["arch_specifier"]
    else:
        raise ValueError(f"Unsupported config.json format in {config_path}")

    checkpoint_file = checkpoint_path / "checkpoints" / "latest-checkpoint.pt"
    if not checkpoint_file.exists():
        checkpoint_candidates = sorted((checkpoint_path / "checkpoints").glob("*.pt")) if (checkpoint_path / "checkpoints").exists() else []
        if checkpoint_candidates:
            checkpoint_file = checkpoint_candidates[-1]
        else:
            raise FileNotFoundError(f"Could not find a Prismatic checkpoint under {checkpoint_path / 'checkpoints'}")

    vision_backbone, _ = get_vision_backbone_and_transform(vision_backbone_id, image_resize_strategy)
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        llm_backbone_id,
        llm_max_length=llm_max_length,
        inference_mode=True,
    )
    action_tokenizer = ActionTokenizer(tokenizer)

    model = AVAVLA.from_pretrained(
        checkpoint_file,
        model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=arch_specifier,
        freeze_weights=True,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
        latent_dim=avavla_cfg.get("latent_dim", 512),
        obs_dim=avavla_cfg.get("obs_dim", 768),
        reasoning_hidden_dim=avavla_cfg.get("reasoning_hidden_dim", 1024),
        transition_hidden_dim=avavla_cfg.get("transition_hidden_dim", 1024),
        exit_gate_hidden_dim=avavla_cfg.get("exit_gate_hidden_dim", 256),
        value_hidden_dim=avavla_cfg.get("value_hidden_dim", 512),
        update_dim=avavla_cfg.get("update_dim", 64),
        reasoning_policy_type=avavla_cfg.get("reasoning_policy_type", "softmax"),
        max_reasoning_steps=max_reasoning_steps,
        exit_threshold=exit_threshold,
        enable_latent_reasoning=enable_latent_reasoning,
    )

    # Load AVA-VLA specific components. Prefer the compact full AVA state; fall back to legacy per-module files.
    avavla_state_path = _find_component_checkpoint(checkpoint_path, "avavla")
    if avavla_state_path is not None:
        print(f"Loading AVA-VLA components from {avavla_state_path}")
        model.load_avavla_state_dict(torch.load(avavla_state_path, map_location=device), strict=True)
    else:
        avavla_components = [
            "reasoning_policy",
            "latent_transition",
            "exit_gate",
            "value_function",
        ]

        for component_name in avavla_components:
            component_path = _find_component_checkpoint(checkpoint_path, component_name)
            if component_path is not None:
                print(f"Loading {component_name} from {component_path}")
                state_dict = torch.load(component_path, map_location=device)
                getattr(model, component_name).load_state_dict(state_dict)
            else:
                print(f"Warning: {component_name} checkpoint not found, using random initialization")

    action_head = None
    action_head_path = _find_component_checkpoint(checkpoint_path, "action_head")
    if avavla_cfg.get("use_l1_regression", False) or action_head_path is not None:
        if action_head_path is not None:
            action_head = L1RegressionActionHead(
                input_dim=model.llm_dim,
                hidden_dim=model.llm_dim,
                action_dim=ACTION_DIM,
            )
            action_head.load_state_dict(torch.load(action_head_path, map_location=device))
            action_head = action_head.to(device)
            action_head.eval()
            model.action_head = action_head
            print(f"Loading action_head from {action_head_path}")
        else:
            print("Warning: AVA config expects L1 action head but no action_head checkpoint was found")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"AVA-VLA model loaded successfully on {device}")
    print(f"  - Latent reasoning: {enable_latent_reasoning}")
    print(f"  - Max reasoning steps: {max_reasoning_steps}")
    print(f"  - Exit threshold: {exit_threshold}")
    
    return model, None, action_tokenizer, norm_stats


def predict_action(
    model: AVAVLA,
    processor,
    image: Image,
    instruction: str,
    unnorm_key: Optional[str] = None,
    num_reasoning_steps: Optional[int] = None,
    device: str = "cuda",
    history_states: Optional[torch.Tensor] = None,
    update_history: bool = True,
) -> tuple:
    """
    Predict action using AVA-VLA.
    
    Args:
        model: AVA-VLA model
        processor: Input processor
        image: PIL Image
        instruction: Task instruction
        unnorm_key: Dataset name for unnormalization
        num_reasoning_steps: Number of reasoning steps (None for adaptive)
        device: Device to run on
    
    Returns:
        actions: Predicted actions
        reasoning_info: Dictionary with reasoning information
    """
    action_head = getattr(model, "action_head", None)
    if action_head is not None:
        return _predict_action_with_l1_head(
            model=model,
            action_head=action_head,
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            num_reasoning_steps=num_reasoning_steps,
            history_states=history_states,
            update_history=update_history,
        )

    with torch.no_grad():
        actions, reasoning_info = model.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            num_reasoning_steps=num_reasoning_steps,
            return_reasoning_info=True,
            history_states=history_states,
            update_history=update_history,
        )

    return actions, reasoning_info


@torch.no_grad()
def _predict_action_with_l1_head(
    model: AVAVLA,
    action_head: L1RegressionActionHead,
    image: Image,
    instruction: str,
    unnorm_key: Optional[str] = None,
    num_reasoning_steps: Optional[int] = None,
    history_states: Optional[torch.Tensor] = None,
    update_history: bool = True,
) -> tuple:
    """Predict continuous actions through the same L1 action head used during fine-tuning."""
    image_transform, tokenizer = model.vision_backbone.image_transform, model.llm_backbone.tokenizer

    prompt_builder = model.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    prompt_text = prompt_builder.get_prompt()

    prompt_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(model.device)
    if isinstance(tokenizer, LlamaTokenizerFast):
        if not torch.all(prompt_input_ids[:, -1] == 29871):
            prompt_input_ids = torch.cat(
                (
                    prompt_input_ids,
                    torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(prompt_input_ids.device),
                ),
                dim=1,
            )
    else:
        raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

    prompt_attention_mask = torch.ones_like(prompt_input_ids, device=prompt_input_ids.device)
    labels = torch.full_like(prompt_input_ids, IGNORE_INDEX)

    action_placeholder = torch.full(
        (prompt_input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK),
        model.action_tokenizer.action_token_begin_idx + 1,
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    stop_token = torch.full(
        (prompt_input_ids.shape[0], 1),
        STOP_INDEX,
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    input_ids = torch.cat([prompt_input_ids, action_placeholder, stop_token], dim=-1)
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    labels = torch.cat([labels, action_placeholder.clone(), stop_token.clone()], dim=-1)

    pixel_values = image_transform(image)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(model.device)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(model.device) for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    if model.enable_latent_reasoning:
        history_states = model._resolve_history_state(history_states, prompt_input_ids.shape[0], prompt_input_ids.device)
        obs_encoding = model.encode_observation(
            pixel_values,
            prompt_input_ids,
            attention_mask=prompt_attention_mask,
            history_states=history_states,
        )
        z_0 = model.initial_latent_proj(obs_encoding)
        final_z, exit_scores, reasoning_info = model.latent_reasoning_forward(
            z_0,
            obs_encoding,
            num_steps=num_reasoning_steps,
            training=False,
            return_trajectory=False,
        )
    else:
        final_z = None
        exit_scores = prompt_input_ids.new_zeros(prompt_input_ids.shape[0], 0, dtype=torch.float32)
        reasoning_info = {"num_steps_performed": 0}

    autocast_enabled = model.device.type == "cuda"
    with torch.autocast("cuda", dtype=model.llm_backbone.half_precision_dtype, enabled=autocast_enabled):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            latent_state=final_z,
            output_hidden_states=True,
            zero_action_token_embeddings=True,
        )

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, model.vision_backbone.num_patches : -1]
    action_token_labels = labels[:, 1:]
    current_action_mask = get_current_action_mask(action_token_labels)
    next_actions_mask = get_next_actions_mask(action_token_labels)
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(input_ids.shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(next(action_head.parameters()).dtype)
    )
    normalized_actions = action_head.predict_action(actions_hidden_states).float().cpu().numpy()[0]
    actions = _unnormalize_actions(model, normalized_actions, unnorm_key)

    reasoning_info["exit_threshold"] = model.exit_threshold
    reasoning_info["exit_scores_history"] = exit_scores.detach().cpu().numpy()
    if "num_steps_per_sample" in reasoning_info and torch.is_tensor(reasoning_info["num_steps_per_sample"]):
        reasoning_info["num_steps_per_sample"] = reasoning_info["num_steps_per_sample"].detach().cpu().numpy()
    if update_history and final_z is not None:
        model._cached_history_state = final_z.detach()
    return actions, reasoning_info


def batch_predict(
    model: AVAVLA,
    processor,
    images: list,
    instructions: list,
    unnorm_key: Optional[str] = None,
    num_reasoning_steps: Optional[int] = None,
    device: str = "cuda",
    stateful_history: bool = False,
) -> tuple:
    """
    Predict actions for a batch of inputs.
    
    Args:
        model: AVA-VLA model
        processor: Input processor
        images: List of PIL Images
        instructions: List of task instructions
        unnorm_key: Dataset name for unnormalization
        num_reasoning_steps: Number of reasoning steps (None for adaptive)
        device: Device to run on
    
    Returns:
        actions: List of predicted actions
        reasoning_infos: List of reasoning information dictionaries
    """
    actions_list = []
    reasoning_infos_list = []
    
    for image, instruction in zip(images, instructions):
        actions, reasoning_info = predict_action(
            model=model,
            processor=processor,
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            num_reasoning_steps=num_reasoning_steps,
            device=device,
            update_history=stateful_history,
        )
        actions_list.append(actions)
        reasoning_infos_list.append(reasoning_info)
    
    return actions_list, reasoning_infos_list


def compute_efficiency_metrics(reasoning_infos: list, max_reasoning_steps: Optional[int] = None) -> Dict:
    """
    Compute efficiency metrics from reasoning information.
    
    Args:
        reasoning_infos: List of reasoning information dictionaries
    
    Returns:
        metrics: Dictionary of efficiency metrics
    """
    num_samples = len(reasoning_infos)
    
    # Extract reasoning steps
    steps_performed = [info.get('num_steps_performed', 0) for info in reasoning_infos]
    exit_scores_history = [info.get('exit_scores_history', []) for info in reasoning_infos]
    
    # Compute statistics
    avg_steps = float(np.mean(steps_performed)) if steps_performed else 0.0
    std_steps = float(np.std(steps_performed)) if steps_performed else 0.0
    min_steps = int(np.min(steps_performed)) if steps_performed else 0
    max_steps = int(np.max(steps_performed)) if steps_performed else 0
    
    # Compute exit score statistics
    final_exit_scores = []
    for exit_scores in exit_scores_history:
        exit_scores = np.asarray(exit_scores)
        if exit_scores.size > 0:
            final_exit_scores.append(float(exit_scores.reshape(-1)[-1]))
    
    avg_exit_score = float(np.mean(final_exit_scores)) if final_exit_scores else 0.0
    early_exit_limit = max_reasoning_steps if max_reasoning_steps is not None else max_steps
    
    metrics = {
        "num_samples": num_samples,
        "avg_reasoning_steps": avg_steps,
        "std_reasoning_steps": std_steps,
        "min_reasoning_steps": min_steps,
        "max_reasoning_steps": max_steps,
        "avg_final_exit_score": avg_exit_score,
        "early_exit_rate": float(np.mean(np.array(steps_performed) < early_exit_limit)) if steps_performed else 0.0,
    }
    
    return metrics


def main():
    """Example usage of AVA-VLA deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy AVA-VLA for inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--instruction", type=str, required=True, help="Task instruction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--enable-latent-reasoning", action="store_true", default=True,
                       help="Enable latent reasoning")
    parser.add_argument("--max-reasoning-steps", type=int, default=None,
                       help="Maximum number of reasoning steps")
    parser.add_argument("--exit-threshold", type=float, default=None,
                       help="Exit threshold for early-exit")
    parser.add_argument("--fixed-steps", type=int, default=None,
                       help="Fixed number of reasoning steps (None for adaptive)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading AVA-VLA model from {args.checkpoint}")
    model, processor, action_tokenizer, norm_stats = load_avavla_model(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        enable_latent_reasoning=args.enable_latent_reasoning,
        max_reasoning_steps=args.max_reasoning_steps,
        exit_threshold=args.exit_threshold,
    )
    
    # Load image
    image = Image.open(args.image).convert("RGB")
    
    # Predict action
    print(f"\nPredicting action for instruction: {args.instruction}")
    actions, reasoning_info = predict_action(
        model=model,
        processor=processor,
        image=image,
        instruction=args.instruction,
        num_reasoning_steps=args.fixed_steps,
        device=args.device,
    )
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Predicted actions: {actions}")
    print(f"Reasoning steps performed: {reasoning_info['num_steps_performed']}")
    print(f"Exit threshold: {reasoning_info['exit_threshold']}")
    
    if reasoning_info['exit_scores_history'] is not None:
        print(f"Exit scores history: {reasoning_info['exit_scores_history'].flatten()}")


if __name__ == "__main__":
    main()
