"""
finetune_avavla.py

Fine-tunes AVA-VLA with latent reasoning, RL-based denoising, and early-exit mechanisms.
"""

import os
import json
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import wandb

from experiments.robot.openvla_utils import (
    model_is_on_hf_hub,
)
from prismatic.conf import ModelConfig
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlas.avavla import AVAVLA
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class AVAVLAFinetuneConfig:
    # fmt: off
    vla_path: str = "path/to/prismatic-openvla-run"  # Local Prismatic/OpenVLA run dir or checkpoint file
    
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs_avavla")
    shuffle_buffer_size: int = 100_000
    
    # AVA-VLA specific
    use_l1_regression: bool = True
    latent_dim: int = 512
    obs_dim: int = 768
    update_dim: int = 64
    reasoning_policy_type: str = "softmax"
    reasoning_hidden_dim: int = 1024
    transition_hidden_dim: int = 1024
    exit_gate_hidden_dim: int = 256
    value_hidden_dim: int = 512
    max_reasoning_steps: int = 5
    exit_threshold: float = 0.8
    enable_latent_reasoning: bool = True
    
    # RL Training
    use_rl_denoising: bool = True
    rl_lr: float = 1e-4
    gamma: float = 0.99
    ppo_clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64
    entropy_coef: float = 0.01
    smoothness_coef: float = 0.1
    value_coef: float = 0.5
    exit_loss_coef: float = 0.1
    reward_error_scale: float = 0.25
    trajectory_reward_scale: float = 0.25
    task_reward_weight: float = 1.0
    action_proxy_reward_weight: float = 1.0
    trajectory_reward_weight: float = 0.25
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100_000
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    val_batches: int = 50
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    history_window_size: int = 2
    
    train_base_vla: bool = False
    
    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "ava-vla"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    
    # fmt: on

def get_run_id(cfg) -> str:
    """Generates an identifier string for an experiment run."""
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.enable_latent_reasoning:
            run_id += f"+latent-dim{cfg.latent_dim}"
        if cfg.use_rl_denoising:
            run_id += "+rl"
        if cfg.train_base_vla:
            run_id += "+train-base"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """Wrap a module with DistributedDataParallel."""
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """Counts and prints the number of trainable parameters in a module."""
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def distributed_barrier() -> None:
    """Synchronize distributed workers when torch.distributed is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def move_to_device(batch_value, device_id: int):
    """Move tensor or nested tensor dicts to the target device."""
    if isinstance(batch_value, dict):
        return {key: move_to_device(value, device_id) for key, value in batch_value.items()}
    return batch_value.to(device_id)


def cast_floating_tensors(batch_value, dtype: torch.dtype):
    """Cast floating tensor leaves while preserving nested pixel-value structure."""
    if isinstance(batch_value, dict):
        return {key: cast_floating_tensors(value, dtype) for key, value in batch_value.items()}
    return batch_value.to(dtype) if torch.is_floating_point(batch_value) else batch_value


def get_batch_history_states(batch: Dict, device_id: int) -> Optional[torch.Tensor]:
    """Return history features h_{t-1} when the RLDS window contains past actions/proprio."""
    history_states = batch.get("history_states")
    if history_states is None or history_states.numel() == 0:
        return None

    history_states = history_states.to(device_id).to(torch.bfloat16)
    history_pad_mask = batch.get("history_pad_mask")
    if history_pad_mask is None:
        return history_states

    mask = history_pad_mask.to(device_id).to(dtype=history_states.dtype).unsqueeze(-1)
    return history_states * mask


def compute_reasoning_rewards(
    batch: Dict,
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    cfg: AVAVLAFinetuneConfig,
    device_id: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build task-level rewards from environment labels when available, plus trajectory consistency."""
    reward_terms = {}
    reward_dtype = ground_truth_actions.dtype

    if "task_rewards" in batch:
        task_reward = batch["task_rewards"].to(device_id).float().clamp(0.0, 1.0)
        reward_terms["task_reward"] = task_reward
        reward = cfg.task_reward_weight * task_reward
        reward_source = "env"
    else:
        action_error = F.l1_loss(predicted_actions, ground_truth_actions, reduction="none")
        action_error = action_error.flatten(start_dim=1).mean(dim=1).float()
        reward_scale = max(float(cfg.reward_error_scale), 1e-6)
        action_proxy_reward = torch.exp(-action_error / reward_scale)
        reward_terms["action_proxy_reward"] = action_proxy_reward
        reward = cfg.action_proxy_reward_weight * action_proxy_reward
        reward_source = "action_proxy"

    if predicted_actions.shape[1] > 1 and cfg.trajectory_reward_weight > 0:
        predicted_delta = predicted_actions[:, 1:].float() - predicted_actions[:, :-1].float()
        expert_delta = ground_truth_actions[:, 1:].float() - ground_truth_actions[:, :-1].float()
        trajectory_error = F.l1_loss(predicted_delta, expert_delta, reduction="none").flatten(start_dim=1).mean(dim=1)
        trajectory_scale = max(float(cfg.trajectory_reward_scale), 1e-6)
        trajectory_reward = torch.exp(-trajectory_error / trajectory_scale)
        reward_terms["trajectory_reward"] = trajectory_reward
        reward = reward + cfg.trajectory_reward_weight * trajectory_reward

    total_weight = 0.0
    if "task_reward" in reward_terms:
        total_weight += cfg.task_reward_weight
    if "action_proxy_reward" in reward_terms:
        total_weight += cfg.action_proxy_reward_weight
    if "trajectory_reward" in reward_terms:
        total_weight += cfg.trajectory_reward_weight
    reward = (reward / max(total_weight, 1e-6)).clamp(0.0, 1.0).to(dtype=reward_dtype)

    metrics = {
        "mean_reward": float(reward.detach().float().mean().cpu()),
        "reward_source_env_rate": 1.0 if reward_source == "env" else 0.0,
    }
    for name, tensor in reward_terms.items():
        metrics[f"mean_{name}"] = float(tensor.detach().float().mean().cpu())
    return reward, metrics


def detach_ppo_rollout(reasoning_info: Dict, rewards: torch.Tensor, exit_targets: torch.Tensor) -> Dict:
    """Freeze one on-policy reasoning rollout for clipped PPO updates."""
    rollout = {}
    for key, value in reasoning_info.items():
        rollout[key] = value.detach() if torch.is_tensor(value) else value
    if "action_log_probs" in rollout:
        rollout["old_action_log_probs"] = rollout["action_log_probs"].detach()
    rollout["ppo_rewards"] = rewards.detach()
    rollout["ppo_exit_targets"] = exit_targets.detach()
    return rollout


def slice_ppo_rollout(rollout: Dict, indices: torch.Tensor) -> Dict:
    """Select a batch minibatch from a frozen PPO rollout."""
    batch_size = rollout["ppo_rewards"].shape[0]
    minibatch = {}
    for key, value in rollout.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            minibatch[key] = value.index_select(0, indices)
        else:
            minibatch[key] = value
    return minibatch


def average_metric_dicts(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Average numeric metric rows emitted by PPO minibatch updates."""
    if not metric_rows:
        return {}
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in metric_rows:
        for name, value in row.items():
            if isinstance(value, (int, float)):
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def run_ppo_updates(
    avavla,
    optimizer,
    trainable_params,
    cfg: AVAVLAFinetuneConfig,
    rollout: Dict,
) -> Dict[str, float]:
    """Run multi-epoch minibatch PPO updates against a frozen on-policy rollout."""
    rewards = rollout["ppo_rewards"]
    exit_targets = rollout["ppo_exit_targets"]
    batch_size = int(rewards.shape[0])
    minibatch_size = max(1, min(int(cfg.ppo_minibatch_size), batch_size))
    metric_rows: List[Dict[str, float]] = []

    for epoch in range(max(1, int(cfg.ppo_epochs))):
        permutation = torch.randperm(batch_size, device=rewards.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            minibatch = slice_ppo_rollout(rollout, indices)
            rl_loss, rl_metrics = avavla.module.compute_rl_loss(
                minibatch,
                minibatch["ppo_rewards"],
                gamma=cfg.gamma,
                entropy_coef=cfg.entropy_coef,
                smoothness_coef=cfg.smoothness_coef,
                value_coef=cfg.value_coef,
                exit_loss_coef=cfg.exit_loss_coef,
                exit_targets=minibatch["ppo_exit_targets"],
                ppo_clip_ratio=cfg.ppo_clip_ratio,
                gae_lambda=cfg.gae_lambda,
                recompute_policy=True,
            )
            optimizer.zero_grad(set_to_none=True)
            rl_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            rl_metrics["ppo_epoch"] = float(epoch + 1)
            rl_metrics["ppo_minibatch_size"] = float(indices.numel())
            rl_metrics["ppo_grad_norm"] = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            metric_rows.append(rl_metrics)

    optimizer.zero_grad(set_to_none=True)
    return average_metric_dicts(metric_rows)


def make_avavla_config(cfg: AVAVLAFinetuneConfig) -> Dict:
    """Serialize AVA-VLA hyperparameters needed to reconstruct the model and action path."""
    return {
        "use_l1_regression": cfg.use_l1_regression,
        "latent_dim": cfg.latent_dim,
        "obs_dim": cfg.obs_dim,
        "update_dim": cfg.update_dim,
        "reasoning_policy_type": cfg.reasoning_policy_type,
        "reasoning_hidden_dim": cfg.reasoning_hidden_dim,
        "transition_hidden_dim": cfg.transition_hidden_dim,
        "exit_gate_hidden_dim": cfg.exit_gate_hidden_dim,
        "value_hidden_dim": cfg.value_hidden_dim,
        "max_reasoning_steps": cfg.max_reasoning_steps,
        "exit_threshold": cfg.exit_threshold,
        "enable_latent_reasoning": cfg.enable_latent_reasoning,
        "rl": {
            "gamma": cfg.gamma,
            "ppo_clip_ratio": cfg.ppo_clip_ratio,
            "gae_lambda": cfg.gae_lambda,
            "ppo_epochs": cfg.ppo_epochs,
            "ppo_minibatch_size": cfg.ppo_minibatch_size,
            "entropy_coef": cfg.entropy_coef,
            "smoothness_coef": cfg.smoothness_coef,
            "value_coef": cfg.value_coef,
            "exit_loss_coef": cfg.exit_loss_coef,
            "reward_error_scale": cfg.reward_error_scale,
            "trajectory_reward_scale": cfg.trajectory_reward_scale,
            "task_reward_weight": cfg.task_reward_weight,
            "action_proxy_reward_weight": cfg.action_proxy_reward_weight,
            "trajectory_reward_weight": cfg.trajectory_reward_weight,
        },
        "history_window_size": cfg.history_window_size,
    }


def _checkpoint_sort_key(path: Path) -> Tuple[int, object]:
    """Sort component checkpoints by numeric step when present, otherwise by mtime/name."""
    suffix = path.name.split("--", 1)[-1].split("_checkpoint.pt", 1)[0]
    if suffix.isdigit():
        return (0, int(suffix))
    try:
        return (1, path.stat().st_mtime)
    except OSError:
        return (1, path.name)


def _find_component_checkpoint(run_dir: Path, stem: str, step: Optional[int] = None) -> Optional[Path]:
    """Find a latest or step-specific component checkpoint."""
    if step is None:
        latest = run_dir / f"{stem}--latest_checkpoint.pt"
        if latest.exists():
            return latest
        candidates = sorted(run_dir.glob(f"{stem}--*_checkpoint.pt"), key=_checkpoint_sort_key)
    else:
        candidates = sorted(run_dir.glob(f"{stem}--{step}_checkpoint.pt"), key=_checkpoint_sort_key)
        if not candidates:
            candidates = sorted(run_dir.glob(f"{stem}--*{step}*_checkpoint.pt"), key=_checkpoint_sort_key)

    return candidates[-1] if candidates else None


def _select_prismatic_checkpoint(run_dir: Path, resume_step: Optional[int] = None) -> Path:
    """Resolve a Prismatic checkpoint file from a run directory or checkpoint file path."""
    if run_dir.is_file():
        return run_dir

    checkpoint_dir = run_dir / "checkpoints"
    if resume_step is None:
        latest = checkpoint_dir / "latest-checkpoint.pt"
        if latest.exists():
            return latest
        candidates = sorted(checkpoint_dir.glob("*.pt"))
    else:
        candidates = sorted(checkpoint_dir.glob(f"*step-{resume_step:06d}*.pt"))
        if not candidates:
            candidates = sorted(checkpoint_dir.glob(f"*{resume_step}*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No Prismatic checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def _read_prismatic_config(vla_path: Path) -> Tuple[dict, Path]:
    """Read local Prismatic/OpenVLA config metadata used by the local AVAVLA wrapper."""
    if vla_path.is_file() and vla_path.parent.name == "checkpoints":
        run_dir = vla_path.parents[1]
    elif vla_path.is_file():
        run_dir = vla_path.parent
    else:
        run_dir = vla_path
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config.json at {config_path}. "
            "The local AVAVLA training path expects a Prismatic/OpenVLA checkpoint directory, not a raw HF model."
        )

    with open(config_path, "r") as f:
        config_json = json.load(f)

    if "vla" in config_json:
        model_cfg = ModelConfig.get_choice_class(config_json["vla"]["base_vlm"])()
        resolved = {
            "model_id": model_cfg.model_id,
            "vision_backbone_id": model_cfg.vision_backbone_id,
            "image_resize_strategy": model_cfg.image_resize_strategy,
            "llm_backbone_id": model_cfg.llm_backbone_id,
            "llm_max_length": model_cfg.llm_max_length,
            "arch_specifier": model_cfg.arch_specifier,
        }
    elif "model" in config_json:
        model_cfg = config_json["model"]
        resolved = {
            "model_id": model_cfg["model_id"],
            "vision_backbone_id": model_cfg["vision_backbone_id"],
            "image_resize_strategy": model_cfg["image_resize_strategy"],
            "llm_backbone_id": model_cfg["llm_backbone_id"],
            "llm_max_length": model_cfg.get("llm_max_length", 2048),
            "arch_specifier": model_cfg["arch_specifier"],
        }
    else:
        raise ValueError(
            f"Unsupported config format in {config_path}. Expected a top-level 'vla' or 'model' section."
        )

    return resolved, config_path


def _component_root(vla_path: Path) -> Path:
    """Resolve the directory that contains AVA/action/training component checkpoints."""
    if vla_path.is_file() and vla_path.parent.name == "checkpoints":
        return vla_path.parents[1]
    if vla_path.is_file():
        return vla_path.parent
    return vla_path


def apply_saved_avavla_config(cfg: AVAVLAFinetuneConfig) -> Dict:
    """Load saved AVA hyperparameters during resume so architecture and action path match the checkpoint."""
    if not cfg.resume:
        return {}

    config_path = _component_root(Path(cfg.vla_path)) / "avavla_config.json"
    if not config_path.exists():
        print(f"Warning: resume requested but no avavla_config.json found at {config_path}; using CLI/default AVA config.")
        return {}

    with open(config_path, "r") as f:
        saved_cfg = json.load(f)

    top_level_fields = (
        "use_l1_regression",
        "latent_dim",
        "obs_dim",
        "update_dim",
        "reasoning_policy_type",
        "reasoning_hidden_dim",
        "transition_hidden_dim",
        "exit_gate_hidden_dim",
        "value_hidden_dim",
        "max_reasoning_steps",
        "exit_threshold",
        "enable_latent_reasoning",
        "history_window_size",
    )
    for field_name in top_level_fields:
        if field_name in saved_cfg:
            setattr(cfg, field_name, saved_cfg[field_name])

    for field_name, value in saved_cfg.get("rl", {}).items():
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, value)

    print(f"Loaded AVA hyperparameters from {config_path}")
    return saved_cfg


def build_avavla_from_prismatic_checkpoint(cfg: AVAVLAFinetuneConfig, device_id: int):
    """Instantiate AVAVLA from a local Prismatic/OpenVLA checkpoint layout."""
    vla_path = Path(cfg.vla_path)
    model_cfg, source_config_path = _read_prismatic_config(vla_path)
    checkpoint_path = _select_prismatic_checkpoint(vla_path, cfg.resume_step if cfg.resume else None)

    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg["llm_max_length"],
        inference_mode=False,
    )
    action_tokenizer = ActionTokenizer(tokenizer)

    avavla = AVAVLA.from_pretrained(
        checkpoint_path,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=False,
        norm_stats={},
        action_tokenizer=action_tokenizer,
        latent_dim=cfg.latent_dim,
        obs_dim=cfg.obs_dim,
        reasoning_hidden_dim=cfg.reasoning_hidden_dim,
        transition_hidden_dim=cfg.transition_hidden_dim,
        exit_gate_hidden_dim=cfg.exit_gate_hidden_dim,
        value_hidden_dim=cfg.value_hidden_dim,
        update_dim=cfg.update_dim,
        reasoning_policy_type=cfg.reasoning_policy_type,
        max_reasoning_steps=cfg.max_reasoning_steps,
        exit_threshold=cfg.exit_threshold,
        enable_latent_reasoning=cfg.enable_latent_reasoning,
    )

    if not cfg.train_base_vla:
        for name, param in avavla.named_parameters():
            param.requires_grad = AVAVLA.is_avavla_parameter_name(name)

    return avavla.to(device_id), tokenizer, image_transform, action_tokenizer, source_config_path, checkpoint_path


def save_base_prismatic_checkpoint(avavla: AVAVLA, checkpoint_path: Path) -> None:
    """Save a Prismatic-compatible checkpoint for the base VLA modules."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {
                "vision_backbone": avavla.vision_backbone.state_dict(),
                "llm_backbone": avavla.llm_backbone.state_dict(),
                "projector": avavla.projector.state_dict(),
            }
        },
        checkpoint_path,
    )


def load_resume_state(
    cfg: AVAVLAFinetuneConfig,
    avavla,
    action_head,
    optimizer,
    scheduler,
    device_id: int,
) -> int:
    """Load AVA-VLA/action/optimizer/scheduler state from an AVA checkpoint directory."""
    if not cfg.resume:
        return 0

    root = _component_root(Path(cfg.vla_path))
    step = cfg.resume_step
    map_location = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    avavla_state_path = _find_component_checkpoint(root, "avavla", step)
    if avavla_state_path is not None:
        avavla.module.load_avavla_state_dict(torch.load(avavla_state_path, map_location=map_location), strict=True)
        print(f"Resumed AVA-VLA components from {avavla_state_path}")
    else:
        legacy_components = ("reasoning_policy", "latent_transition", "exit_gate", "value_function")
        loaded_components = []
        for component_name in legacy_components:
            component_path = _find_component_checkpoint(root, component_name, step)
            if component_path is None:
                continue
            getattr(avavla.module, component_name).load_state_dict(torch.load(component_path, map_location=map_location))
            loaded_components.append(component_name)
        if loaded_components:
            print(
                "Resumed legacy AVA component checkpoints "
                f"({', '.join(loaded_components)}); newly added AVA projection modules use current initialization."
            )

    if action_head is not None:
        action_head_path = _find_component_checkpoint(root, "action_head", step)
        if action_head_path is not None:
            action_head.module.load_state_dict(torch.load(action_head_path, map_location=map_location))
            print(f"Resumed action head from {action_head_path}")

    training_state_path = _find_component_checkpoint(root, "training_state", step)
    if training_state_path is None:
        return step or 0

    training_state = torch.load(training_state_path, map_location=map_location)
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    print(f"Resumed optimizer/scheduler from {training_state_path}")
    return int(training_state.get("log_step", step or 0))


def run_forward_pass_with_latent_reasoning(
    avavla,
    action_head,
    batch,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    training: bool = True,
    include_rl_loss: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float], Dict]:
    """
    Compute forward pass with latent reasoning.
    
    Returns:
        loss: Total loss
        metrics: Dictionary of metrics
        reasoning_info: Dictionary with reasoning trajectory information
    """
    metrics = {}
    reasoning_info = {}
    
    # Get ground-truth actions
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    
    # Encode observations
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = move_to_device(batch["pixel_values"], device_id)
    labels = batch["labels"].to(device_id)
    history_states = get_batch_history_states(batch, device_id)

    obs_encoding = avavla.module.encode_observation(
        pixel_values,
        input_ids,
        attention_mask=attention_mask,
        history_states=history_states,
    )
    
    # Initialize latent state
    if cfg.enable_latent_reasoning:
        z_t = avavla.module.initial_latent_proj(obs_encoding)
    else:
        z_t = torch.zeros(obs_encoding.shape[0], cfg.latent_dim, device=obs_encoding.device)
    
    # Perform latent reasoning and collect trajectory
    if cfg.enable_latent_reasoning:
        final_z, exit_scores, info = avavla.module.latent_reasoning_forward(
            z_t,
            obs_encoding,
            num_steps=cfg.max_reasoning_steps,
            training=training,
            return_trajectory=True,
        )
        reasoning_info.update(info)
        reasoning_info['exit_scores'] = exit_scores
    else:
        final_z = z_t
    
    # Run VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = avavla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=cast_floating_tensors(pixel_values, torch.bfloat16),
            labels=labels,
            latent_state=final_z if cfg.enable_latent_reasoning else None,
            output_hidden_states=True,
            zero_action_token_embeddings=cfg.use_l1_regression,
        )
    
    # Compute action loss
    if cfg.use_l1_regression:
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        
        ground_truth_token_ids = labels[:, 1:]
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
        
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )
        
        predicted_actions = action_head.module.predict_action(actions_hidden_states)
        action_loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
        
        # Compute metrics
        ground_truth_curr_action = ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
        next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
        
        metrics.update({
            "action_loss": action_loss.item(),
            "curr_action_l1_loss": curr_action_l1_loss.item(),
            "next_actions_l1_loss": next_actions_l1_loss.item(),
        })
    else:
        action_loss = output.loss
        metrics["action_loss"] = action_loss.item()
    
    # Compute rewards and optionally the first PPO loss against the just-collected rollout.
    if cfg.enable_latent_reasoning and cfg.use_rl_denoising and training:
        with torch.no_grad():
            if cfg.use_l1_regression:
                reward, reward_metrics = compute_reasoning_rewards(
                    batch=batch,
                    predicted_actions=predicted_actions,
                    ground_truth_actions=ground_truth_actions,
                    cfg=cfg,
                    device_id=device_id,
                )
            else:
                reward = batch["task_rewards"].to(device_id).to(dtype=ground_truth_actions.dtype) if "task_rewards" in batch else torch.ones(ground_truth_actions.shape[0], device=device_id, dtype=ground_truth_actions.dtype)
                reward_metrics = {
                    "mean_reward": float(reward.detach().float().mean().cpu()),
                    "reward_source_env_rate": 1.0 if "task_rewards" in batch else 0.0,
                }
            exit_targets = (reward >= cfg.exit_threshold).to(dtype=reward.dtype)

        if "action_log_probs" in reasoning_info:
            reasoning_info["old_action_log_probs"] = reasoning_info["action_log_probs"].detach()
        reasoning_info["ppo_rewards"] = reward.detach()
        reasoning_info["ppo_exit_targets"] = exit_targets.detach()

        if include_rl_loss:
            rl_loss, rl_loss_info = avavla.module.compute_rl_loss(
                reasoning_info,
                reward,
                gamma=cfg.gamma,
                entropy_coef=cfg.entropy_coef,
                smoothness_coef=cfg.smoothness_coef,
                value_coef=cfg.value_coef,
                exit_loss_coef=cfg.exit_loss_coef,
                exit_targets=exit_targets,
                ppo_clip_ratio=cfg.ppo_clip_ratio,
                gae_lambda=cfg.gae_lambda,
                recompute_policy=True,
            )
            metrics.update(rl_loss_info)
        else:
            rl_loss = torch.tensor(0.0, device=device_id)
        metrics.update(reward_metrics)
        metrics["exit_target_rate"] = float(exit_targets.detach().float().mean().cpu())
    else:
        rl_loss = torch.tensor(0.0, device=device_id)
    
    # Total loss
    total_loss = action_loss + rl_loss
    metrics["total_loss"] = total_loss.item()
    
    return total_loss, metrics, reasoning_info


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    avavla,
    source_config_path,
    action_head,
    train_dataset,
    optimizer,
    scheduler,
    distributed_state,
    avavla_config,
) -> None:
    """Save all training checkpoints."""
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir / "checkpoints", exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        shutil.copy2(source_config_path, checkpoint_dir / "config.json")
        with open(checkpoint_dir / "avavla_config.json", "w") as f:
            json.dump(avavla_config, f, indent=2)
        print(f"Saving Model Checkpoint for Step {log_step}")
    
    distributed_barrier()
    
    if distributed_state.is_main_process:
        # Keep a Prismatic-compatible base checkpoint so deployment can reconstruct the full AVA-VLA model.
        save_base_prismatic_checkpoint(avavla.module, checkpoint_dir / "checkpoints" / "latest-checkpoint.pt")
        
        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.module.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
        
        # Save AVA-VLA specific components
        if cfg.enable_latent_reasoning:
            torch.save(
                avavla.module.get_avavla_state_dict(),
                checkpoint_dir / f"avavla--{checkpoint_name_suffix}",
            )
            torch.save(avavla.module.reasoning_policy.state_dict(), 
                      checkpoint_dir / f"reasoning_policy--{checkpoint_name_suffix}")
            torch.save(avavla.module.latent_transition.state_dict(), 
                      checkpoint_dir / f"latent_transition--{checkpoint_name_suffix}")
            torch.save(avavla.module.exit_gate.state_dict(), 
                      checkpoint_dir / f"exit_gate--{checkpoint_name_suffix}")
            torch.save(avavla.module.value_function.state_dict(), 
                      checkpoint_dir / f"value_function--{checkpoint_name_suffix}")
        torch.save(
            {
                "log_step": log_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / f"training_state--{checkpoint_name_suffix}",
        )
    
    distributed_barrier()
    

def run_validation(
    avavla,
    action_head,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
) -> Dict[str, float]:
    """Run a bounded validation pass over an iterable validation dataloader."""
    was_training = avavla.training
    avavla.eval()
    if action_head is not None:
        action_head.eval()

    totals: Dict[str, float] = {}
    count = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in val_dataloader:
            if count >= cfg.val_batches or (time.time() - start_time) > cfg.val_time_limit:
                break
            _, metrics, _ = run_forward_pass_with_latent_reasoning(
                avavla=avavla,
                action_head=action_head,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                cfg=cfg,
                num_patches=num_patches,
                training=False,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            count += 1

    if was_training:
        avavla.train()
        if action_head is not None:
            action_head.train()

    if count == 0:
        return {}
    return {name: value / count for name, value in totals.items()}


@draccus.wrap()
def finetune_avavla(cfg: AVAVLAFinetuneConfig) -> None:
    """Fine-tunes AVA-VLA on demonstration dataset."""
    cfg.vla_path = cfg.vla_path.rstrip("/")
    apply_saved_avavla_config(cfg)
    print(f"Fine-tuning AVA-VLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
    
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}"
    )
    
    # Download or load model
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path

    avavla, tokenizer, image_transform, action_tokenizer, source_config_path, source_checkpoint_path = (
        build_avavla_from_prismatic_checkpoint(cfg, device_id)
    )
    avavla_config = make_avavla_config(cfg)
    
    # Set number of images in VLA input when using the HF-style vision backbone.
    if hasattr(avavla.vision_backbone, "set_num_images_in_input"):
        avavla.vision_backbone.set_num_images_in_input(1)
    
    avavla = wrap_ddp(avavla, device_id, find_unused=True)
    
    # Action head
    if cfg.use_l1_regression:
        action_head = wrap_ddp(
            L1RegressionActionHead(
                input_dim=avavla.module.llm_dim,
                hidden_dim=avavla.module.llm_dim,
                action_dim=ACTION_DIM
            ).to(torch.bfloat16).to(device_id),
            device_id
        )
    else:
        action_head = None
    
    # Get number of patches
    num_patches = avavla.module.vision_backbone.num_patches
    
    # Optimizer
    trainable_params = [param for param in avavla.parameters() if param.requires_grad]
    if action_head is not None:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    
    # Separate optimizer groups for the base VLA, AVA reasoning modules, and the action head.
    vla_params = [
        param for name, param in avavla.module.named_parameters()
        if param.requires_grad and not AVAVLA.is_avavla_parameter_name(name)
    ]
    avavla_params = [
        param for name, param in avavla.module.named_parameters()
        if param.requires_grad and AVAVLA.is_avavla_parameter_name(name)
    ]

    optimizer_groups = []
    if vla_params:
        optimizer_groups.append({'params': vla_params, 'lr': cfg.learning_rate})
    if avavla_params:
        optimizer_groups.append({'params': avavla_params, 'lr': cfg.rl_lr})
    if action_head is not None:
        action_head_params = [param for param in action_head.parameters() if param.requires_grad]
        if action_head_params:
            optimizer_groups.append({'params': action_head_params, 'lr': cfg.learning_rate})
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters found for AVA-VLA fine-tuning.")
    for param_group in optimizer_groups:
        param_group["initial_lr"] = param_group["lr"]

    optimizer = AdamW(optimizer_groups)
    
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    start_step = load_resume_state(cfg, avavla, action_head, optimizer, scheduler, device_id)
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,
        use_proprio=False,
    )
    
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(avavla.module.vision_backbone.default_image_resolution[-2:]),
        window_size=cfg.history_window_size,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    avavla.module.norm_stats = train_dataset.dataset_statistics
    
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        shutil.copy2(source_config_path, run_dir / "config.json")
        with open(run_dir / "avavla_config.json", "w") as f:
            json.dump(avavla_config, f, indent=2)
    
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    val_dataloader = None
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(avavla.module.vision_backbone.default_image_resolution[-2:]),
            window_size=cfg.history_window_size,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            train=False,
            image_aug=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )
    
    # Metrics tracking
    recent_metrics = {
        "total_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }
    if cfg.use_rl_denoising:
        recent_metrics.update({
            "total_rl_loss": deque(maxlen=cfg.grad_accumulation_steps),
            "policy_loss": deque(maxlen=cfg.grad_accumulation_steps),
            "value_loss": deque(maxlen=cfg.grad_accumulation_steps),
            "exit_loss": deque(maxlen=cfg.grad_accumulation_steps),
            "mean_reward": deque(maxlen=cfg.grad_accumulation_steps),
            "mean_task_reward": deque(maxlen=cfg.grad_accumulation_steps),
            "mean_action_proxy_reward": deque(maxlen=cfg.grad_accumulation_steps),
            "mean_trajectory_reward": deque(maxlen=cfg.grad_accumulation_steps),
            "reward_source_env_rate": deque(maxlen=cfg.grad_accumulation_steps),
            "exit_target_rate": deque(maxlen=cfg.grad_accumulation_steps),
            "ppo_ratio_mean": deque(maxlen=cfg.grad_accumulation_steps),
            "ppo_clip_fraction": deque(maxlen=cfg.grad_accumulation_steps),
            "ppo_minibatch_size": deque(maxlen=cfg.grad_accumulation_steps),
        })
    
    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, initial=start_step, leave=False) as progress:
        avavla.train()
        optimizer.zero_grad()
        log_step = start_step
        pending_ppo_rollouts: List[Dict] = []
        
        for batch_idx, batch in enumerate(dataloader):
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} already reached! Stopping training...")
                break

            loss, metrics, reasoning_info = run_forward_pass_with_latent_reasoning(
                avavla=avavla,
                action_head=action_head,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                cfg=cfg,
                num_patches=num_patches,
                training=True,
                include_rl_loss=False,
            )
            if (
                cfg.enable_latent_reasoning
                and cfg.use_rl_denoising
                and "ppo_rewards" in reasoning_info
                and "ppo_exit_targets" in reasoning_info
            ):
                pending_ppo_rollouts.append(
                    detach_ppo_rollout(
                        reasoning_info,
                        reasoning_info["ppo_rewards"],
                        reasoning_info["ppo_exit_targets"],
                    )
                )
            
            # Normalize loss
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # LR warmup
            if 0 < cfg.lr_warmup_steps and log_step < cfg.lr_warmup_steps:
                lr_progress = min((log_step + 1) / cfg.lr_warmup_steps, 1.0)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["initial_lr"] * (0.1 + 0.9 * lr_progress)
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if pending_ppo_rollouts:
                    ppo_metric_rows = [
                        run_ppo_updates(avavla, optimizer, trainable_params, cfg, rollout)
                        for rollout in pending_ppo_rollouts
                    ]
                    metrics.update(average_metric_dicts(ppo_metric_rows))
                    pending_ppo_rollouts.clear()

                # Store metrics after both the supervised update and the PPO rollout updates finish.
                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                log_step += 1
                progress.update()

                # Logging
                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    smoothened_metrics = {}
                    for name, deque_obj in recent_metrics.items():
                        if deque_obj and len(deque_obj) > 0:
                            smoothened_metrics[name] = sum(deque_obj) / len(deque_obj)

                    log_dict = {
                        f"Train/{name.replace('_', ' ').title()}": value
                        for name, value in smoothened_metrics.items()
                    }

                    # Add reasoning statistics
                    if reasoning_info:
                        log_dict["Train/Avg Steps Performed"] = reasoning_info.get('num_steps_performed', 0)

                    wandb.log(log_dict, step=log_step)
                    wandb.log(
                        {
                            "Train/Learning Rate": scheduler.get_last_lr()[0],
                            "Train/Grad Norm": float(grad_norm),
                        },
                        step=log_step,
                    )

                if cfg.use_val_set and val_dataloader is not None and log_step % cfg.val_freq == 0:
                    val_metrics = run_validation(
                        avavla=avavla,
                        action_head=action_head,
                        val_dataloader=val_dataloader,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        cfg=cfg,
                        num_patches=num_patches,
                    )
                    if distributed_state.is_main_process and val_metrics:
                        wandb.log(
                            {f"Val/{name.replace('_', ' ').title()}": value for name, value in val_metrics.items()},
                            step=log_step,
                        )

                # Save checkpoint
                if log_step % cfg.save_freq == 0:
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=log_step,
                        avavla=avavla,
                        source_config_path=source_config_path,
                        action_head=action_head,
                        train_dataset=train_dataset,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        distributed_state=distributed_state,
                        avavla_config=avavla_config,
                    )

                # Stop condition
                if log_step >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break


if __name__ == "__main__":
    finetune_avavla()
