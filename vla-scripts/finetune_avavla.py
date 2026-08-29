"""
finetune_avavla.py

Fine-tunes AVA-VLA with latent reasoning, RL-based denoising, and early-exit mechanisms.
"""

import os
import json
import math
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb
from PIL import Image

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
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    STOP_INDEX,
    NormalizationType,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CHECKPOINT_IMPLEMENTATION_VERSION = 9


@dataclass
class AVAVLAFinetuneConfig:
    # fmt: off
    vla_path: str = "path/to/prismatic-openvla-run"  # Local Prismatic/OpenVLA run dir or checkpoint file
    llm_config_path: Path = Path("models/llama2-7b-ms-tokenizer")
    
    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs_avavla")
    shuffle_buffer_size: int = 2_048
    seed: int = 0
    use_wrist_image: bool = True
    use_proprio: bool = True
    proprio_dim: int = 8
    
    # AVA-VLA specific
    use_l1_regression: bool = True
    latent_dim: int = 512
    obs_dim: int = 768
    update_dim: int = 64
    reasoning_policy_type: str = "gaussian"
    reasoning_hidden_dim: int = 512
    reasoning_num_heads: int = 8
    reasoning_num_layers: int = 4
    transition_hidden_dim: int = 1024
    exit_gate_hidden_dim: int = 256
    value_hidden_dim: int = 512
    action_hidden_dim: int = 2048
    dropout: float = 0.1
    max_reasoning_steps: int = 5
    exit_threshold: float = 0.55
    enable_latent_reasoning: bool = True
    
    # RL Training
    use_rl_denoising: bool = True
    policy_lr: float = 3e-5
    critic_lr: float = 1e-4
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
    ppo_effective_batch_size: int = 512
    ppo_environment_steps: int = 1_200_000
    ppo_rollout_size_per_rank: int = 64
    ppo_optimizer_steps: Optional[int] = None
    online_ppo: bool = True
    online_envs_per_rank: int = 8
    online_num_steps_wait: int = 10
    online_env_img_res: int = 256
    online_center_crop: bool = True
    libero_task_suites: str = "libero_spatial,libero_object,libero_goal,libero_10"
    bc_steps: int = 100_000
    latent_warmup_steps: int = 50_000
    exit_calibration_steps: int = 10_000
    require_online_task_rewards: bool = True
    exit_calibration_lookahead: int = 3
    exit_calibration_delta: float = 0.05
    exit_calibration_buffer_rollouts: int = 128
    ppo_bc_aux_coef: float = 1.0
    action_ppo_std: float = 0.05
    action_ppo_coef: float = 1.0
    
    require_384px_backbone: bool = False
    require_dinosiglip_backbone: bool = True
    require_robot_pretrained_base: bool = True
    # Training configuration
    batch_size: int = 8  # Per device; 8 GPUs reproduce the paper's global BC batch size of 64.
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    lr_warmup_steps: int = 0
    num_steps_before_decay: Optional[int] = None
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    max_steps: Optional[int] = None  # Derived from 1.2M interactions and the actual distributed batch size.
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    val_batches: int = 50
    save_freq: int = 1_000
    ppo_checkpoint_interval_updates: int = 10
    exit_checkpoint_interval_steps: int = 1_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    history_window_size: int = NUM_ACTIONS_CHUNK + 1
    
    train_base_vla: bool = False
    
    # Logging
    use_wandb: bool = False
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


def resolve_paper_training_budget(cfg: AVAVLAFinetuneConfig, world_size: int) -> None:
    """Resolve all four paper stages and count true robot-environment interactions."""
    world_size = max(1, int(world_size))
    if not cfg.enable_latent_reasoning:
        raise ValueError("Paper reproduction requires AVA latent reasoning to be enabled.")
    if not cfg.use_l1_regression:
        raise ValueError("Paper reproduction requires the OpenVLA-OFT continuous L1 action head.")
    if cfg.max_reasoning_steps <= 0:
        raise ValueError("max_reasoning_steps must be positive for AVA latent reasoning.")
    if cfg.history_window_size < NUM_ACTIONS_CHUNK + 1:
        raise ValueError(
            "history_window_size must cover the previous policy decision: "
            f"expected at least {NUM_ACTIONS_CHUNK + 1}."
        )
    if cfg.online_envs_per_rank <= 0 or cfg.ppo_rollout_size_per_rank <= 0:
        raise ValueError("Online PPO environment and rollout sizes must be positive.")
    if cfg.shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be positive.")
    if cfg.save_freq <= 0:
        raise ValueError("save_freq must be positive so resumable checkpoints are produced.")
    if cfg.ppo_checkpoint_interval_updates <= 0:
        raise ValueError("ppo_checkpoint_interval_updates must be positive.")
    if cfg.exit_checkpoint_interval_steps <= 0:
        raise ValueError("exit_checkpoint_interval_steps must be positive.")
    if cfg.action_ppo_std <= 0:
        raise ValueError("action_ppo_std must be positive for the stochastic Stage-3 action policy.")
    if cfg.action_ppo_coef <= 0:
        raise ValueError("action_ppo_coef must be positive for joint Stage-3 PPO.")
    effective_rollout_batch = cfg.ppo_rollout_size_per_rank * world_size
    if effective_rollout_batch != cfg.ppo_effective_batch_size:
        raise ValueError(
            "PPO rollout batch mismatch: "
            f"{cfg.ppo_rollout_size_per_rank} samples/rank * {world_size} ranks = "
            f"{effective_rollout_batch}, expected {cfg.ppo_effective_batch_size}."
        )
    if cfg.ppo_minibatch_size % world_size != 0:
        raise ValueError("Global PPO mini-batch size must be divisible by world size.")
    if cfg.ppo_rollout_size_per_rank % cfg.online_envs_per_rank != 0:
        raise ValueError(
            "ppo_rollout_size_per_rank must be divisible by online_envs_per_rank."
        )

    # Each policy query executes one full action chunk. The online runner also
    # tracks the exact global env.step count and stops at the requested budget.
    interactions_per_update = effective_rollout_batch * NUM_ACTIONS_CHUNK
    if cfg.ppo_optimizer_steps is None:
        cfg.ppo_optimizer_steps = math.ceil(cfg.ppo_environment_steps / interactions_per_update)
    expected_total = (
        cfg.bc_steps
        + cfg.latent_warmup_steps
        + cfg.ppo_optimizer_steps
        + cfg.exit_calibration_steps
    )
    if cfg.max_steps is None:
        cfg.max_steps = expected_total
    print(
        f"Paper schedule: BC={cfg.bc_steps}, warmup={cfg.latent_warmup_steps}, "
        f"PPO~={cfg.ppo_optimizer_steps} updates/{cfg.ppo_environment_steps} env steps, "
        f"exit calibration={cfg.exit_calibration_steps}."
    )


def should_checkpoint_ppo(
    cfg: AVAVLAFinetuneConfig,
    ppo_update: int,
    global_env_steps: int,
) -> bool:
    """Checkpoint by PPO update cadence and always at the environment budget boundary."""
    return (
        int(ppo_update) % int(cfg.ppo_checkpoint_interval_updates) == 0
        or int(global_env_steps) >= int(cfg.ppo_environment_steps)
    )


class DistributedModuleProxy(nn.Module):
    """Expose the `.module` interface while using explicit stable gradient averaging."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> nn.Module:
    """Synchronize random initialization and wrap without PyTorch's unstable reducer."""
    if dist.is_available() and dist.is_initialized():
        for parameter in module.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
    return DistributedModuleProxy(module)


def count_parameters(module: nn.Module, name: str) -> None:
    """Counts and prints the number of trainable parameters in a module."""
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def configure_stage_trainability(avavla, action_head, stage: str, train_base_vla: bool) -> List[nn.Parameter]:
    """Freeze exactly the components that the paper leaves fixed in each stage."""
    model = avavla.module if hasattr(avavla, "module") else avavla
    head = action_head.module if action_head is not None and hasattr(action_head, "module") else action_head
    for parameter in model.parameters():
        parameter.requires_grad = False
    if head is not None:
        for parameter in head.parameters():
            parameter.requires_grad = False

    if stage == "bc":
        bc_prefixes = (
            "projector.",
            "visual_obs_proj.",
            "visual_view_fusion.",
            "text_obs_proj.",
            "proprio_obs_proj.",
            "history_obs_proj.",
            "obs_fusion.",
            "initial_latent_proj.",
            "latent_to_llm.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name == "latent_action_scale" or name.startswith(bc_prefixes)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad = True
        if train_base_vla:
            for name, parameter in model.named_parameters():
                if not AVAVLA.is_avavla_parameter_name(name):
                    parameter.requires_grad = True
    elif stage == "latent_warmup":
        prefixes = ("reasoning_policy.", "latent_transition.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)
    elif stage == "ppo":
        # Appendix B.3 jointly fine-tunes the multimodal encoder, latent
        # dynamics, critic, and action policy.  The large frozen vision/Llama
        # backbones remain fixed, matching the OpenVLA-OFT setup.
        prefixes = (
            "projector.",
            "visual_obs_proj.",
            "visual_view_fusion.",
            "text_obs_proj.",
            "proprio_obs_proj.",
            "history_obs_proj.",
            "obs_fusion.",
            "initial_latent_proj.",
            "reasoning_policy.",
            "latent_transition.",
            "value_function.",
            "latent_to_llm.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name == "latent_action_scale" or name.startswith(prefixes)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad = True
    elif stage == "exit_calibration":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("exit_gate.")
    else:
        raise ValueError(f"Unknown paper training stage: {stage}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if head is not None:
        trainable.extend(parameter for parameter in head.parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError(f"No trainable parameters configured for stage={stage}")
    return trainable


def build_stage_optimizer(
    avavla,
    action_head,
    stage: str,
    cfg: AVAVLAFinetuneConfig,
) -> tuple[Adam, LambdaLR, List[nn.Parameter]]:
    """Build stage-specific Adam groups with the appendix learning rates."""
    trainable = configure_stage_trainability(avavla, action_head, stage, cfg.train_base_vla)
    model = avavla.module if hasattr(avavla, "module") else avavla
    head = action_head.module if action_head is not None and hasattr(action_head, "module") else action_head
    groups = []

    def add_group(parameters, lr: float) -> None:
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "initial_lr": lr})

    add_group(
        (
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("reasoning_policy.")
            and not name.startswith("value_function.")
            and not name.startswith("exit_gate.")
        ),
        cfg.learning_rate,
    )
    add_group(model.reasoning_policy.parameters(), cfg.policy_lr)
    add_group(model.value_function.parameters(), cfg.critic_lr)
    add_group(model.exit_gate.parameters(), cfg.critic_lr)
    if head is not None:
        add_group(head.parameters(), cfg.learning_rate)

    optimizer = Adam(groups, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)
    def lr_scale(step: int) -> float:
        warmup_scale = (
            min(1.0, float(step + 1) / float(cfg.lr_warmup_steps))
            if cfg.lr_warmup_steps > 0
            else 1.0
        )
        decay_scale = (
            0.1
            if cfg.num_steps_before_decay is not None and step >= cfg.num_steps_before_decay
            else 1.0
        )
        return warmup_scale * decay_scale

    scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
    return optimizer, scheduler, trainable


def distributed_barrier() -> None:
    """Synchronize distributed workers when torch.distributed is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def synchronize_gradients(parameters: List[nn.Parameter], chunk_elements: int = 16_777_216) -> None:
    """Average contiguous gradient chunks, matching data-parallel DDP semantics."""
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    for parameter in parameters:
        source_gradient = parameter.grad
        # All ranks execute identical stage graphs; a missing gradient therefore
        # means the parameter is unused on every rank for this objective.
        if source_gradient is None:
            continue
        gradient = source_gradient.contiguous()
        flat_gradient = gradient.view(-1)
        for chunk in flat_gradient.split(chunk_elements):
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
            chunk.div_(world_size)
        if gradient.data_ptr() != source_gradient.data_ptr():
            source_gradient.copy_(gradient)



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


def get_batch_history_states(
    avavla,
    batch: Dict,
    device_id: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Encode the preceding RLDS observation into the same latent space used online."""
    history_states = batch.get("history_states")
    if history_states is not None and history_states.numel() > 0:
        history_states = history_states.to(device_id).to(torch.bfloat16)
        history_pad_mask = batch.get("history_pad_mask")
        if history_pad_mask is None:
            return history_states
        mask = history_pad_mask.to(device_id).to(dtype=history_states.dtype)
        while mask.dim() < history_states.dim():
            mask = mask.unsqueeze(-1)
        return history_states * mask

    history_pixels = batch.get("pixel_values_history")
    if history_pixels is None:
        return None
    history_pixels = move_to_device(history_pixels, device_id)
    wrist_history = batch.get("pixel_values_wrist_history")
    if wrist_history is not None:
        wrist_history = move_to_device(wrist_history, device_id)
    proprio_history = batch.get("proprio_history")
    if proprio_history is not None:
        proprio_history = proprio_history.to(device_id).to(torch.bfloat16)

    model = avavla.module if hasattr(avavla, "module") else avavla
    history_obs = model.encode_observation(
        cast_floating_tensors(history_pixels, torch.bfloat16),
        input_ids,
        attention_mask=attention_mask,
        pixel_values_wrist=(
            cast_floating_tensors(wrist_history, torch.bfloat16)
            if wrist_history is not None
            else None
        ),
        proprio=proprio_history,
        labels=labels,
    )
    history_latent = model.initial_latent_proj(history_obs)
    history_pad_mask = batch.get("history_pad_mask")
    if history_pad_mask is not None:
        history_latent = history_latent * history_pad_mask.to(
            device_id,
            dtype=history_latent.dtype,
        ).unsqueeze(-1)
    return history_latent


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


def reduce_online_rollout_metrics(
    metrics: Dict[str, float],
    device_id: int,
) -> Dict[str, float]:
    """Return true global episode counts/rates while averaging per-query rewards."""
    reduced = dict(metrics)
    if not (dist.is_available() and dist.is_initialized()):
        return reduced

    count_names = sorted(
        name
        for name in metrics
        if name.endswith("_completed_episodes") or name.endswith("_completed_successes")
    )
    counts = torch.tensor(
        [float(metrics[name]) for name in count_names],
        device=device_id,
        dtype=torch.float64,
    )
    mean_reward = torch.tensor(
        float(metrics.get("online_mean_reward", 0.0)),
        device=device_id,
        dtype=torch.float64,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(mean_reward, op=dist.ReduceOp.SUM)
    mean_reward.div_(dist.get_world_size())

    for name, value in zip(count_names, counts.cpu().tolist()):
        reduced[name] = value
    prefixes = {
        name[: -len("_completed_episodes")]
        for name in count_names
        if name.endswith("_completed_episodes")
    }
    for prefix in prefixes:
        episodes = reduced.get(f"{prefix}_completed_episodes", 0.0)
        successes = reduced.get(f"{prefix}_completed_successes", 0.0)
        reduced[f"{prefix}_success_rate"] = successes / episodes if episodes else 0.0
    reduced["online_mean_reward"] = float(mean_reward.cpu())
    return reduced


@torch.no_grad()
def prepare_temporal_ppo_targets(
    avavla,
    rollout: Dict,
    cfg: AVAVLAFinetuneConfig,
) -> Dict[str, float]:
    """Compute frozen GAE targets across latent steps and successive env queries."""
    model = avavla.module if hasattr(avavla, "module") else avavla
    # Keep GAE in FP32 even though rollout storage is BF16. The value head is
    # also FP32, so this makes both kernel dtypes and long-horizon statistics
    # explicit rather than depending on whichever autocast scope called us.
    states = rollout["latent_states"].float()
    next_states = rollout["next_latent_states"].float()
    valid = rollout.get("valid_mask", states.new_ones(states.shape[:2])).float()
    value_dtype = states.dtype
    value_parameters = getattr(model.value_function, "parameters", None)
    if callable(value_parameters):
        first_parameter = next(iter(value_parameters()), None)
        if first_parameter is not None and torch.is_floating_point(first_parameter):
            value_dtype = first_parameter.dtype
    values = model.value_function(states.to(dtype=value_dtype)).squeeze(-1).detach().float()
    entropies = rollout["action_entropies"].float()
    smoothness = (next_states - states).pow(2).mean(dim=-1)
    rewards = rollout["ppo_rewards"].float()
    dones = rollout["ppo_dones"].bool()
    if "ppo_chunk_lengths" not in rollout:
        raise RuntimeError(
            "PPO rollout is missing ppo_chunk_lengths; semi-Markov discounts would be incorrect."
        )
    chunk_lengths = rollout["ppo_chunk_lengths"].long().clamp_min(1)
    env_ids = rollout["ppo_env_ids"].long()
    time_ids = rollout["ppo_time_indices"].long()
    bootstrap = rollout["ppo_bootstrap_values"].to(device=states.device).float()

    task_rewards = torch.zeros_like(values)
    last_valid = valid.sum(dim=1).long().clamp_min(1) - 1
    task_rewards.scatter_(1, last_valid.unsqueeze(1), rewards.unsqueeze(1))
    immediate = task_rewards - cfg.entropy_coef * entropies - cfg.smoothness_coef * smoothness
    advantages = torch.zeros_like(values)

    for env_id in range(int(rollout["ppo_num_envs"])):
        row_indices = torch.nonzero(env_ids == env_id, as_tuple=False).squeeze(-1)
        row_indices = row_indices[torch.argsort(time_ids.index_select(0, row_indices), descending=True)]
        next_value = bootstrap[env_id]
        next_advantage = states.new_zeros(())
        for row_index in row_indices.tolist():
            valid_steps = int(valid[row_index].sum().item())
            for latent_step in reversed(range(valid_steps)):
                at_query_end = latent_step == valid_steps - 1
                if at_query_end:
                    # One policy query can execute 1..NUM_ACTIONS_CHUNK real
                    # environment transitions. Treat it as a semi-Markov step
                    # so sparse success credit is discounted by elapsed robot
                    # time rather than by an assumed single transition.
                    elapsed = int(chunk_lengths[row_index].item())
                    value_discount = float(cfg.gamma) ** elapsed
                    gae_discount = (float(cfg.gamma) * float(cfg.gae_lambda)) ** elapsed
                else:
                    value_discount = float(cfg.gamma)
                    gae_discount = float(cfg.gamma) * float(cfg.gae_lambda)
                continuation = (
                    (~dones[row_index]).to(dtype=states.dtype)
                    if at_query_end
                    else states.new_ones(())
                )
                following_value = next_value if at_query_end else values[row_index, latent_step + 1]
                delta = (
                    immediate[row_index, latent_step]
                    + value_discount * continuation * following_value
                    - values[row_index, latent_step]
                )
                next_advantage = (
                    delta
                    + gae_discount * continuation * next_advantage
                )
                advantages[row_index, latent_step] = next_advantage
            next_value = values[row_index, 0]
            next_advantage = advantages[row_index, 0]

    local_sum = (advantages * valid).sum()
    local_sq_sum = (advantages.square() * valid).sum()
    local_count = valid.sum()
    moments = torch.stack([local_sum, local_sq_sum, local_count])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(moments, op=dist.ReduceOp.SUM)
    mean = moments[0] / moments[2].clamp_min(1.0)
    variance = (moments[1] / moments[2].clamp_min(1.0) - mean.square()).clamp_min(1e-8)
    normalized_advantages = (advantages - mean) / variance.sqrt()

    rollout["ppo_advantages"] = normalized_advantages.detach()
    rollout["ppo_returns"] = (advantages + values).detach()
    rollout["ppo_step_rewards"] = immediate.detach()
    return {
        "gae_raw_mean": float(mean.float().cpu()),
        "gae_raw_std": float(variance.sqrt().float().cpu()),
        "bootstrap_value_mean": float(bootstrap.float().mean().cpu()),
        "mean_action_chunk_length": float(chunk_lengths.float().mean().cpu()),
        "terminal_fraction": float(dones.float().mean().cpu()),
    }


_ACTION_PIXEL_PREFIX = "ppo_action_pixel_values__"


def _reconstruct_ppo_action_pixels(rollout: Dict):
    """Restore tensor/dict image-transform outputs saved by online collection."""
    if "ppo_action_pixel_values" in rollout:
        return rollout["ppo_action_pixel_values"]
    names = sorted(
        key[len(_ACTION_PIXEL_PREFIX) :]
        for key in rollout
        if key.startswith(_ACTION_PIXEL_PREFIX)
    )
    if not names:
        raise RuntimeError("Joint PPO rollout is missing replayable OFT pixel inputs")
    return {name: rollout[f"{_ACTION_PIXEL_PREFIX}{name}"] for name in names}


def recompute_robot_action_features(
    avavla,
    rollout: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the current z -> OpenVLA-OFT -> action-feature path with gradients."""
    from vla_scripts.online_policy import recompute_robot_action_features as shared_recompute
    return shared_recompute(avavla, rollout)

    model = avavla.module if hasattr(avavla, "module") else avavla
    feature_names = ("primary_visual", "wrist_visual", "text", "proprio", "history")
    required = [
        "update_actions",
        "ppo_action_input_ids",
        "ppo_action_attention_mask",
        "ppo_action_labels",
        *[f"ppo_observation_{name}" for name in feature_names],
    ]
    missing = [name for name in required if name not in rollout]
    if missing:
        raise RuntimeError(f"Joint PPO rollout is missing end-to-end action replay fields: {missing}")

    observation_features = {
        name: rollout[f"ppo_observation_{name}"]
        for name in feature_names
    }
    pixel_values = _reconstruct_ppo_action_pixels(rollout)
    input_ids = rollout["ppo_action_input_ids"]
    attention_mask = rollout["ppo_action_attention_mask"]
    labels = rollout["ppo_action_labels"]
    device = input_ids.device
    autocast_dtype = model.llm_backbone.half_precision_dtype

    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=device.type == "cuda",
    ):
        obs_encoding = model.fuse_observation_features(observation_features)
        current_z = model.initial_latent_proj(obs_encoding)
        update_actions = rollout["update_actions"].to(
            device=current_z.device,
            dtype=current_z.dtype,
        )
        valid_mask = rollout.get(
            "valid_mask",
            current_z.new_ones(update_actions.shape[:2]),
        ).to(device=current_z.device)
        for step in range(update_actions.shape[1]):
            proposed_z = model.latent_transition(
                current_z,
                obs_encoding,
                update_actions[:, step],
            )
            current_z = torch.where(
                valid_mask[:, step].bool().unsqueeze(-1),
                proposed_z.to(dtype=current_z.dtype),
                current_z,
            )

        output = avavla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=cast_floating_tensors(pixel_values, autocast_dtype),
            labels=labels,
            latent_state=current_z,
            output_hidden_states=True,
            zero_action_token_embeddings=True,
        )

    shifted_labels = labels[:, 1:]
    action_mask = (
        get_current_action_mask(shifted_labels)
        | get_next_actions_mask(shifted_labels)
    )
    text_hidden_states = output.hidden_states[-1][
        :, model.vision_backbone.num_patches : -1
    ]
    batch_size = input_ids.shape[0]
    action_features = text_hidden_states[action_mask].reshape(
        batch_size,
        NUM_ACTIONS_CHUNK * ACTION_DIM,
        -1,
    )
    return action_features, current_z


def compute_robot_action_ppo_loss(
    avavla,
    action_head,
    rollout: Dict,
    clip_ratio: float,
    action_policy_std: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """PPO surrogate for the sampled OpenVLA-OFT continuous action chunk."""
    required = (
        "ppo_robot_action_samples",
        "ppo_robot_action_means",
        "ppo_old_robot_action_log_probs",
        "ppo_advantages",
        "valid_mask",
    )
    missing = [name for name in required if name not in rollout]
    if missing:
        raise RuntimeError(f"Joint PPO rollout is missing robot-action fields: {missing}")
    if action_head is None:
        raise RuntimeError("Joint Stage-3 PPO requires the continuous action head")

    action_features, current_final_z = recompute_robot_action_features(avavla, rollout)
    head_module = action_head.module if hasattr(action_head, "module") else action_head
    action_dtype = next(head_module.parameters()).dtype
    current_means = action_head(action_features.to(dtype=action_dtype)).float()
    samples = rollout["ppo_robot_action_samples"].to(
        device=current_means.device,
        dtype=current_means.dtype,
    )
    distribution = torch.distributions.Normal(
        current_means,
        torch.full_like(current_means, float(action_policy_std)),
    )
    current_log_probs = distribution.log_prob(samples).sum(dim=(-1, -2))
    old_log_probs = rollout["ppo_old_robot_action_log_probs"].to(
        device=current_means.device,
        dtype=current_means.dtype,
    ).detach()
    valid = rollout["valid_mask"].to(device=current_means.device)
    last_valid = valid.sum(dim=1).long().clamp_min(1) - 1
    query_advantages = rollout["ppo_advantages"].to(
        device=current_means.device,
        dtype=current_means.dtype,
    ).gather(1, last_valid.unsqueeze(1)).squeeze(1).detach()
    ratio = torch.exp((current_log_probs - old_log_probs).clamp(-20.0, 20.0))
    unclipped = ratio * query_advantages
    clipped = torch.clamp(
        ratio,
        1.0 - clip_ratio,
        1.0 + clip_ratio,
    ) * query_advantages
    action_policy_loss = -torch.minimum(unclipped, clipped).mean()

    old_means = rollout["ppo_robot_action_means"].to(
        device=current_means.device,
        dtype=current_means.dtype,
    )
    sample_deviation = (samples - old_means).pow(2).mean().sqrt()
    mean_shift = (current_means - old_means).pow(2).mean().sqrt()
    out_of_bounds_fraction = (current_means.abs() > 1.0).float().mean()
    return action_policy_loss, {
        "robot_action_policy_loss": float(action_policy_loss.detach().cpu()),
        "robot_action_ppo_ratio_mean": float(ratio.mean().detach().cpu()),
        "robot_action_ppo_clip_fraction": float(
            ((ratio - 1.0).abs() > clip_ratio).float().mean().detach().cpu()
        ),
        "robot_action_sample_deviation": float(sample_deviation.detach().cpu()),
        "robot_action_mean_shift": float(mean_shift.detach().cpu()),
        "robot_action_mean_out_of_bounds_fraction": float(
            out_of_bounds_fraction.detach().cpu()
        ),
        "robot_action_final_latent_norm": float(
            current_final_z.float().norm(dim=-1).mean().detach().cpu()
        ),
        "robot_action_distribution_std": float(action_policy_std),
        "ppo_action_path_recomputed": 1.0,
        "ppo_joint_action_update": 1.0,
    }


def _module_gradient_norm(module: nn.Module) -> float:
    """Return the local pre-reduction L2 norm for one PPO component."""
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


def run_ppo_updates(
    avavla,
    action_head,
    optimizer,
    scheduler,
    trainable_params,
    cfg: AVAVLAFinetuneConfig,
    rollout: Dict,
) -> Dict[str, float]:
    """Run multi-epoch minibatch PPO updates against a frozen on-policy rollout."""
    model = avavla.module if hasattr(avavla, "module") else avavla
    head_module = action_head.module if action_head is not None and hasattr(action_head, "module") else action_head
    if head_module is None:
        raise RuntimeError("Paper Stage-3 PPO requires an action head")

    # Collection is in eval mode. Replay the whole current OFT action path in
    # eval mode as well so the first likelihood ratio is exactly on-policy while
    # retaining gradients through projector -> frozen Llama -> action head.
    prior_model_training = model.training
    prior_head_training = head_module.training
    model.eval()
    head_module.eval()
    rewards = rollout["ppo_rewards"]
    batch_size = int(rewards.shape[0])
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    # Paper mini-batch 64 is global: on 8 ranks every DDP backward sees 8 samples/rank.
    local_minibatch_size = max(1, int(cfg.ppo_minibatch_size) // world_size)
    minibatch_size = min(local_minibatch_size, batch_size)
    metric_rows: List[Dict[str, float]] = []

    for epoch in range(max(1, int(cfg.ppo_epochs))):
        permutation = torch.randperm(batch_size, device=rewards.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            minibatch = slice_ppo_rollout(rollout, indices)
            optimizer.zero_grad(set_to_none=True)
            rl_loss, rl_metrics = avavla(
                training_objective="ppo",
                reasoning_trajectories=minibatch,
                rl_rewards=minibatch["ppo_rewards"],
                objective_kwargs={
                    "gamma": cfg.gamma,
                    "entropy_coef": cfg.entropy_coef,
                    "smoothness_coef": cfg.smoothness_coef,
                    "value_coef": cfg.value_coef,
                    "exit_loss_coef": 0.0,
                    "ppo_clip_ratio": cfg.ppo_clip_ratio,
                    "gae_lambda": cfg.gae_lambda,
                    "recompute_policy": True,
                    "recompute_dynamics": True,
                    "recompute_observation": True,
                    "train_exit_gate": False,
                },
            )
            action_rl_loss, action_rl_metrics = compute_robot_action_ppo_loss(
                avavla,
                action_head,
                minibatch,
                clip_ratio=cfg.ppo_clip_ratio,
                action_policy_std=cfg.action_ppo_std,
            )
            joint_ppo_loss = rl_loss + cfg.action_ppo_coef * action_rl_loss
            joint_ppo_loss.backward()
            action_rl_metrics.update(
                {
                    "ppo_projector_grad_norm": _module_gradient_norm(model.projector),
                    "ppo_latent_to_llm_grad_norm": _module_gradient_norm(model.latent_to_llm),
                    "ppo_action_head_grad_norm": _module_gradient_norm(head_module),
                }
            )
            synchronize_gradients(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            rl_metrics["latent_rl_loss"] = float(rl_loss.detach().cpu())
            rl_metrics.update(action_rl_metrics)
            rl_metrics["joint_ppo_loss"] = float(joint_ppo_loss.detach().cpu())
            rl_metrics["total_rl_loss"] = float(joint_ppo_loss.detach().cpu())
            rl_metrics["ppo_epoch"] = float(epoch + 1)
            rl_metrics["ppo_minibatch_size"] = float(indices.numel())
            rl_metrics["ppo_grad_norm"] = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
            metric_rows.append(rl_metrics)

    optimizer.zero_grad(set_to_none=True)
    model.train(prior_model_training)
    head_module.train(prior_head_training)
    return average_metric_dicts(metric_rows)


def make_avavla_config(cfg: AVAVLAFinetuneConfig) -> Dict:
    """Serialize AVA-VLA hyperparameters needed to reconstruct the model and action path."""
    return {
        "implementation_version": CHECKPOINT_IMPLEMENTATION_VERSION,
        "action_policy_parameterization": "openvla_oft_direct_gaussian_v1",
        "use_l1_regression": cfg.use_l1_regression,
        "latent_dim": cfg.latent_dim,
        "obs_dim": cfg.obs_dim,
        "update_dim": cfg.update_dim,
        "reasoning_policy_type": cfg.reasoning_policy_type,
        "reasoning_hidden_dim": cfg.reasoning_hidden_dim,
        "reasoning_num_heads": cfg.reasoning_num_heads,
        "reasoning_num_layers": cfg.reasoning_num_layers,
        "transition_hidden_dim": cfg.transition_hidden_dim,
        "exit_gate_hidden_dim": cfg.exit_gate_hidden_dim,
        "value_hidden_dim": cfg.value_hidden_dim,
        "action_hidden_dim": cfg.action_hidden_dim,
        "dropout": cfg.dropout,
        "max_reasoning_steps": cfg.max_reasoning_steps,
        "exit_threshold": cfg.exit_threshold,
        "enable_latent_reasoning": cfg.enable_latent_reasoning,
        "proprio_dim": cfg.proprio_dim,
        "use_wrist_image": cfg.use_wrist_image,
        "use_proprio": cfg.use_proprio,
        "require_384px_backbone": cfg.require_384px_backbone,
        "require_dinosiglip_backbone": cfg.require_dinosiglip_backbone,
        "require_robot_pretrained_base": cfg.require_robot_pretrained_base,
        "paper_schedule": {
            "bc_steps": cfg.bc_steps,
            "latent_warmup_steps": cfg.latent_warmup_steps,
            "ppo_environment_steps": cfg.ppo_environment_steps,
            "exit_calibration_steps": cfg.exit_calibration_steps,
        },
        "rl": {
            "policy_lr": cfg.policy_lr,
            "critic_lr": cfg.critic_lr,
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
            "ppo_effective_batch_size": cfg.ppo_effective_batch_size,
            "ppo_environment_steps": cfg.ppo_environment_steps,
            "exit_calibration_lookahead": cfg.exit_calibration_lookahead,
            "exit_calibration_delta": cfg.exit_calibration_delta,
            "exit_calibration_buffer_rollouts": cfg.exit_calibration_buffer_rollouts,
            "ppo_bc_aux_coef": cfg.ppo_bc_aux_coef,
            "action_ppo_std": cfg.action_ppo_std,
            "action_ppo_coef": cfg.action_ppo_coef,
            "ppo_checkpoint_interval_updates": cfg.ppo_checkpoint_interval_updates,
            "regularizers_in_reward_only": True,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": cfg.learning_rate,
            "beta1": cfg.adam_beta1,
            "beta2": cfg.adam_beta2,
            "eps": cfg.adam_eps,
            "max_grad_norm": cfg.max_grad_norm,
            "exit_checkpoint_interval_steps": cfg.exit_checkpoint_interval_steps,
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
    manifest_path = run_dir / "CHECKPOINT_COMPLETE.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest_step = int(manifest.get("log_step", -1))
            if step is None or int(step) == manifest_step:
                matches = [
                    run_dir / relative_name
                    for relative_name in manifest.get("required_files", {})
                    if Path(relative_name).name.startswith(f"{stem}--")
                ]
                if len(matches) == 1 and matches[0].is_file():
                    return matches[0]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    if step is None:
        latest = run_dir / f"{stem}--latest_checkpoint.pt"
        if latest.exists():
            return latest
        candidates = sorted(run_dir.glob(f"{stem}--*_checkpoint.pt"), key=_checkpoint_sort_key)
    else:
        candidates = sorted(run_dir.glob(f"{stem}--{step}_checkpoint.pt"), key=_checkpoint_sort_key)
        if not candidates:
            candidates = sorted(run_dir.glob(f"{stem}--*{step}*_checkpoint.pt"), key=_checkpoint_sort_key)
        if not candidates:
            latest = run_dir / f"{stem}--latest_checkpoint.pt"
            manifest_path = run_dir / "CHECKPOINT_COMPLETE.json"
            if latest.exists() and manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as stream:
                        manifest_step = int(json.load(stream).get("log_step", -1))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    manifest_step = -1
                if manifest_step == int(step):
                    candidates = [latest]

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

    vla_metadata = config_json.get("vla", {})
    metadata_text = json.dumps(vla_metadata, sort_keys=True).lower()
    resolved["robot_pretrained"] = bool(
        "openvla" in resolved["model_id"].lower()
        or (
            str(config_json.get("stage", "")).lower() == "vla-full-train"
            and "oxe" in metadata_text
        )
    )

    return resolved, config_path


def _component_root(vla_path: Path) -> Path:
    """Resolve the directory that contains AVA/action/training component checkpoints."""
    if vla_path.is_file() and vla_path.parent.name == "checkpoints":
        return vla_path.parents[1]
    if vla_path.is_file():
        return vla_path.parent
    return vla_path


def _atomic_torch_save(payload, path: Path) -> None:
    """Write a torch artifact atomically so interruption cannot corrupt the last checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Dict, path: Path) -> None:
    """Atomically replace a JSON metadata file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_if_different(source: Path, destination: Path) -> None:
    """Atomically copy metadata while allowing resume-in-place without SameFileError."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _link_or_copy_base_checkpoint(source: Path, destination: Path) -> None:
    """Publish the immutable robot-pretrained base without rewriting 30 GB per save."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.is_file():
        return
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing source base checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _archive_bc_complete_checkpoint(checkpoint_dir: Path, manifest: Dict) -> Optional[Path]:
    """Keep an independently loadable BC snapshot when latest-only rotation is enabled.

    The main run directory remains the resumable latest checkpoint.  Files in the
    completed-BC generation are hard-linked into a stage-specific directory when
    possible, so later warmup/PPO rotation can unlink the main-directory names
    without deleting the BC weights or duplicating their storage.
    """
    if manifest.get("stage") != "bc_complete":
        return None

    checkpoint_dir = Path(checkpoint_dir)
    archive_dir = checkpoint_dir / "stage_checkpoints" / "bc_complete"
    if archive_dir.is_dir():
        archived_manifest = _validate_checkpoint_manifest(archive_dir)
        if (
            archived_manifest.get("stage") == "bc_complete"
            and int(archived_manifest.get("log_step", -1)) == int(manifest.get("log_step", -2))
        ):
            return archive_dir
        raise RuntimeError(f"Refusing to replace incompatible BC archive: {archive_dir}")

    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = archive_dir.with_name(f".{archive_dir.name}.tmp-{os.getpid()}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        required_files = manifest.get("required_files", {})
        if not isinstance(required_files, dict) or not required_files:
            raise RuntimeError("Cannot archive BC checkpoint without required_files")
        for relative_name, expected_size in required_files.items():
            source = checkpoint_dir / relative_name
            if not source.is_file() or source.stat().st_size != int(expected_size):
                raise RuntimeError(
                    f"Cannot archive incomplete BC artifact {source}: "
                    f"size={source.stat().st_size if source.is_file() else -1}, expected={expected_size}"
                )
            _link_or_copy_base_checkpoint(source, temporary_dir / relative_name)

        _atomic_write_json(manifest, temporary_dir / "CHECKPOINT_COMPLETE.json")
        _atomic_write_json(
            {
                "stage": "bc_complete",
                "log_step": int(manifest["log_step"]),
                "recommended_max_reasoning_steps": 0,
                "source_checkpoint_dir": str(checkpoint_dir.resolve()),
            },
            temporary_dir / "STAGE_CHECKPOINT.json",
        )
        _validate_checkpoint_manifest(temporary_dir)
        os.replace(temporary_dir, archive_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return archive_dir


def _validate_checkpoint_manifest(root: Path) -> Dict:
    """Reject partial or pre-contract checkpoints before loading any training state."""
    manifest_path = Path(root) / "CHECKPOINT_COMPLETE.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing atomic checkpoint manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest.get("implementation_version", 0)) < CHECKPOINT_IMPLEMENTATION_VERSION:
        raise RuntimeError(
            f"Checkpoint {root} uses implementation_version="
            f"{manifest.get('implementation_version', 0)} and is incompatible with "
            "the current action-PPO checkpoint schema."
        )
    required_files = manifest.get("required_files")
    if not isinstance(required_files, dict) or not required_files:
        raise RuntimeError(f"Checkpoint manifest has no required_files map: {manifest_path}")
    for relative_name, expected_size in required_files.items():
        artifact = Path(root) / relative_name
        actual_size = artifact.stat().st_size if artifact.is_file() else -1
        if actual_size != int(expected_size) or actual_size <= 0:
            raise RuntimeError(
                f"Incomplete checkpoint artifact {artifact}: size={actual_size}, expected={expected_size}"
            )
    return manifest


def apply_saved_avavla_config(cfg: AVAVLAFinetuneConfig) -> Dict:
    """Load saved AVA hyperparameters during resume so architecture and action path match the checkpoint."""
    if not cfg.resume:
        return {}

    component_root = _component_root(Path(cfg.vla_path))
    config_path = component_root / "avavla_config.json"
    if not config_path.exists():
        raise RuntimeError(f"Resume requested but no avavla_config.json was found at {config_path}")

    with open(config_path, "r") as f:
        saved_cfg = json.load(f)
    if int(saved_cfg.get("implementation_version", 0)) < CHECKPOINT_IMPLEMENTATION_VERSION:
        raise RuntimeError(
            "This checkpoint predates the end-to-end OpenVLA-OFT action-PPO and direct-action parameterization contract and cannot be resumed safely. "
            "Start a new run from the OpenVLA Prismatic base checkpoint."
        )
    _validate_checkpoint_manifest(component_root)

    top_level_fields = (
        "use_l1_regression",
        "latent_dim",
        "obs_dim",
        "update_dim",
        "reasoning_policy_type",
        "reasoning_hidden_dim",
        "reasoning_num_heads",
        "reasoning_num_layers",
        "transition_hidden_dim",
        "exit_gate_hidden_dim",
        "value_hidden_dim",
        "action_hidden_dim",
        "dropout",
        "max_reasoning_steps",
        "exit_threshold",
        "enable_latent_reasoning",
        "history_window_size",
        "proprio_dim",
        "use_wrist_image",
        "use_proprio",
        "require_384px_backbone",
        "require_dinosiglip_backbone",
        "require_robot_pretrained_base",
    )
    for field_name in top_level_fields:
        if field_name in saved_cfg:
            setattr(cfg, field_name, saved_cfg[field_name])

    for field_name, value in saved_cfg.get("rl", {}).items():
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, value)
    for field_name, value in saved_cfg.get("paper_schedule", {}).items():
        if hasattr(cfg, field_name):
            setattr(cfg, field_name, value)

    optimizer_field_aliases = {
        "beta1": "adam_beta1",
        "beta2": "adam_beta2",
        "eps": "adam_eps",
    }
    for field_name, value in saved_cfg.get("optimizer", {}).items():
        config_field = optimizer_field_aliases.get(field_name, field_name)
        if field_name != "name" and hasattr(cfg, config_field):
            setattr(cfg, config_field, value)

    print(f"Loaded AVA hyperparameters from {config_path}")
    return saved_cfg


def build_avavla_from_prismatic_checkpoint(cfg: AVAVLAFinetuneConfig, device_id: int):
    """Instantiate AVAVLA from a local Prismatic/OpenVLA checkpoint layout."""
    vla_path = Path(cfg.vla_path)
    model_cfg, source_config_path = _read_prismatic_config(vla_path)
    if cfg.require_robot_pretrained_base and not model_cfg["robot_pretrained"]:
        raise ValueError(
            "Paper reproduction must start from the robot-pretrained OpenVLA Prismatic checkpoint "
            "(openvla/openvla-7b-prismatic), not a generic Prism/LLaVA checkpoint."
        )
    # AVA checkpoint directories keep the reconstructed base VLA at a stable
    # latest-checkpoint path. resume_step selects compact AVA/training state,
    # not a nonexistent step-named copy of the 30 GB base checkpoint.
    is_avavla_checkpoint = (_component_root(vla_path) / "avavla_config.json").exists()
    base_step = None if is_avavla_checkpoint else (cfg.resume_step if cfg.resume else None)
    checkpoint_path = _select_prismatic_checkpoint(vla_path, base_step)

    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        initialize_empty=True,
    )
    if not cfg.llm_config_path.is_dir():
        raise FileNotFoundError(f"Missing local Llama config/tokenizer directory: {cfg.llm_config_path}")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg["llm_max_length"],
        inference_mode=False,
        initialize_empty=True,
        config_and_tokenizer_path=str(cfg.llm_config_path),
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
        proprio_dim=cfg.proprio_dim,
        reasoning_hidden_dim=cfg.reasoning_hidden_dim,
        reasoning_num_heads=cfg.reasoning_num_heads,
        reasoning_num_layers=cfg.reasoning_num_layers,
        transition_hidden_dim=cfg.transition_hidden_dim,
        exit_gate_hidden_dim=cfg.exit_gate_hidden_dim,
        value_hidden_dim=cfg.value_hidden_dim,
        update_dim=cfg.update_dim,
        dropout=cfg.dropout,
        reasoning_policy_type=cfg.reasoning_policy_type,
        max_reasoning_steps=cfg.max_reasoning_steps,
        exit_threshold=cfg.exit_threshold,
        enable_latent_reasoning=cfg.enable_latent_reasoning,
    )

    if cfg.require_dinosiglip_backbone and not model_cfg["vision_backbone_id"].startswith("dinosiglip-"):
        raise ValueError(
            "AVA-VLA requires the fused DINOv2+SigLIP vision backbone; "
            f"got {model_cfg['vision_backbone_id']}."
        )
    image_resolution = tuple(int(x) for x in avavla.vision_backbone.default_image_resolution[-2:])
    if cfg.require_384px_backbone and image_resolution != (384, 384):
        raise ValueError(
            f"AVA-VLA paper reproduction requires the 384px DINOv2+SigLIP backbone; got {image_resolution} "
            f"from {model_cfg['vision_backbone_id']}."
        )

    if not cfg.train_base_vla:
        for name, param in avavla.named_parameters():
            param.requires_grad = AVAVLA.is_avavla_parameter_name(name)

    return avavla.to(device_id), tokenizer, image_transform, action_tokenizer, source_config_path, checkpoint_path


def save_base_prismatic_checkpoint(avavla: AVAVLA, checkpoint_path: Path) -> None:
    """Atomically save a Prismatic-compatible checkpoint for base VLA modules."""
    _atomic_torch_save(
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
) -> Dict:
    """Load AVA-VLA/action/optimizer/scheduler state from an AVA checkpoint directory."""
    if not cfg.resume:
        return {"log_step": 0}

    root = _component_root(Path(cfg.vla_path))
    _validate_checkpoint_manifest(root)
    step = cfg.resume_step
    map_location = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    avavla_state_path = _find_component_checkpoint(root, "avavla", step)
    if avavla_state_path is not None:
        avavla.module.load_avavla_state_dict(torch.load(avavla_state_path, map_location=map_location), strict=True)
        print(f"Resumed AVA-VLA components from {avavla_state_path}")
    else:
        raise RuntimeError(
            f"Missing complete AVA-VLA component checkpoint under {root}; refusing random initialization."
        )

    if action_head is not None:
        action_head_path = _find_component_checkpoint(root, "action_head", step)
        if action_head_path is not None:
            action_head.module.load_state_dict(torch.load(action_head_path, map_location=map_location))
            print(f"Resumed action head from {action_head_path}")
        else:
            raise RuntimeError(f"Missing L1 action-head checkpoint under {root}")

    training_state_path = _find_component_checkpoint(root, "training_state", step)
    if training_state_path is None:
        raise RuntimeError(f"Missing optimizer/scheduler training state under {root}")

    training_state = torch.load(training_state_path, map_location=map_location)
    if int(training_state.get("implementation_version", 0)) < CHECKPOINT_IMPLEMENTATION_VERSION:
        raise RuntimeError(
            f"Training state {training_state_path} is incompatible with the current checkpoint schema"
        )
    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        print(f"Resumed optimizer/scheduler from {training_state_path}")
    else:
        print(f"Resumed model state at log step {training_state.get('log_step', step or 0)}")
    training_state["log_step"] = int(training_state.get("log_step", step or 0))

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rng_state_path = _find_component_checkpoint(root, f"rng_state_rank{rank}", step)
    if rng_state_path is None:
        raise RuntimeError(f"Missing RNG state for rank {rank} under {root}")
    rng_state = torch.load(rng_state_path, map_location="cpu")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device_id)
    print(f"Restored RNG state for rank {rank} from {rng_state_path}")
    return training_state


def run_forward_pass_with_latent_reasoning(
    avavla,
    action_head,
    batch,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    training: bool = True,
    include_rl_loss: bool = False,
    use_latent_state: bool = False,
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
    pixel_values_wrist = batch.get("pixel_values_wrist")
    if pixel_values_wrist is not None:
        pixel_values_wrist = move_to_device(pixel_values_wrist, device_id)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device_id).to(torch.bfloat16)
    labels = batch["labels"].to(device_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history_states = get_batch_history_states(
            avavla,
            batch,
            device_id,
            input_ids,
            attention_mask,
            labels,
        )

    final_z = None
    if cfg.enable_latent_reasoning and use_latent_state:
        # The checkpoint is BF16.  Keep the observation encoder and latent
        # rollout under the same autocast contract as the action forward pass;
        # otherwise stage-3 BC auxiliary updates can mix FP32 image inputs with
        # BF16 backbone weights and fail before PPO starts.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            obs_encoding = avavla.module.encode_observation(
                cast_floating_tensors(pixel_values, torch.bfloat16),
                input_ids,
                attention_mask=attention_mask,
                history_states=history_states,
                pixel_values_wrist=(
                    cast_floating_tensors(pixel_values_wrist, torch.bfloat16)
                    if pixel_values_wrist is not None
                    else None
                ),
                proprio=proprio,
                labels=labels,
            )
            z_t = avavla.module.initial_latent_proj(obs_encoding)
            final_z, exit_scores, info = avavla.module.latent_reasoning_forward(
                z_t,
                obs_encoding,
                num_steps=cfg.max_reasoning_steps,
                training=training,
                return_trajectory=True,
            )
        reasoning_info.update(info)
        reasoning_info["exit_scores"] = exit_scores

    # Run VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = avavla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=cast_floating_tensors(pixel_values, torch.bfloat16),
            pixel_values_wrist=(
                cast_floating_tensors(pixel_values_wrist, torch.bfloat16)
                if pixel_values_wrist is not None else None
            ),
            proprio=proprio,
            history_states=history_states,
            labels=labels,
            latent_state=final_z,
            initialize_latent_from_observation=cfg.enable_latent_reasoning and not use_latent_state,
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
        
        predicted_actions = action_head(actions_hidden_states)
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
    
    # Demonstration batches are used only for BC/warmup. PPO rewards are
    # collected from live LIBERO env.step outcomes in the online stage.
    if include_rl_loss:
        raise RuntimeError("Offline RL loss is disabled; use the online LIBERO PPO stage.")
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
    paper_state: Optional[Dict] = None,
) -> None:
    """Save all training checkpoints."""
    stage_name = str((paper_state or {}).get("stage", "checkpoint")).replace("/", "_")
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = f"step-{log_step}-{stage_name}_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"step-{log_step}-{stage_name}_checkpoint.pt"
    previous_required_files = set()
    
    if distributed_state.is_main_process:
        previous_manifest_path = checkpoint_dir / "CHECKPOINT_COMPLETE.json"
        if previous_manifest_path.is_file():
            try:
                previous_required_files = set(json.loads(previous_manifest_path.read_text())["required_files"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                previous_required_files = set()
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir / "checkpoints", exist_ok=True)
        if not (checkpoint_dir / "dataset_statistics.json").exists():
            save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        _copy_if_different(source_config_path, checkpoint_dir / "config.json")
        _atomic_write_json(avavla_config, checkpoint_dir / "avavla_config.json")
        print(f"Saving Model Checkpoint for Step {log_step}")
    
    distributed_barrier()

    rank = int(distributed_state.process_index)
    rng_state_path = checkpoint_dir / f"rng_state_rank{rank}--{checkpoint_name_suffix}"
    _atomic_torch_save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(distributed_state.local_process_index)
            if torch.cuda.is_available()
            else None,
        },
        rng_state_path,
    )
    distributed_barrier()
    
    if distributed_state.is_main_process:
        if cfg.use_l1_regression and action_head is not None:
            _atomic_torch_save(
                action_head.module.state_dict(),
                checkpoint_dir / f"action_head--{checkpoint_name_suffix}",
            )
        
        # Save AVA-VLA specific components
        if cfg.enable_latent_reasoning:
            _atomic_torch_save(
                avavla.module.get_avavla_state_dict(),
                checkpoint_dir / f"avavla--{checkpoint_name_suffix}",
            )
        training_state_path = checkpoint_dir / f"training_state--{checkpoint_name_suffix}"
        _atomic_torch_save(
            {
                "implementation_version": CHECKPOINT_IMPLEMENTATION_VERSION,
                "log_step": log_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "paper_state": dict(paper_state or {}),
            },
            training_state_path,
        )

        required_paths = [
            checkpoint_dir / "config.json",
            checkpoint_dir / "dataset_statistics.json",
            checkpoint_dir / "avavla_config.json",
            checkpoint_dir / "checkpoints" / "latest-checkpoint.pt",
            checkpoint_dir / f"action_head--{checkpoint_name_suffix}",
            checkpoint_dir / f"avavla--{checkpoint_name_suffix}",
            training_state_path,
        ]
        required_paths.extend(
            checkpoint_dir / f"rng_state_rank{rank_index}--{checkpoint_name_suffix}"
            for rank_index in range(int(distributed_state.num_processes))
        )
        missing = [str(path) for path in required_paths if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise RuntimeError(f"Refusing to publish incomplete checkpoint manifest; missing={missing}")
        checkpoint_manifest = {
            "implementation_version": CHECKPOINT_IMPLEMENTATION_VERSION,
            "log_step": int(log_step),
            "stage": (paper_state or {}).get("stage"),
            "required_files": {
                str(path.relative_to(checkpoint_dir)): int(path.stat().st_size)
                for path in required_paths
            },
        }
        _atomic_write_json(checkpoint_manifest, checkpoint_dir / "CHECKPOINT_COMPLETE.json")
        if cfg.save_latest_checkpoint_only:
            archived_bc_dir = _archive_bc_complete_checkpoint(checkpoint_dir, checkpoint_manifest)
            if archived_bc_dir is not None:
                print(f"Preserved completed BC checkpoint at {archived_bc_dir}", flush=True)
            current_required_files = {
                str(path.relative_to(checkpoint_dir)) for path in required_paths
            }
            removable_prefixes = ("action_head--", "avavla--", "training_state--", "rng_state_rank")
            discovered_generation_files = {
                str(path.relative_to(checkpoint_dir))
                for prefix in removable_prefixes
                for path in checkpoint_dir.glob(f"{prefix}*")
            }
            obsolete_files = (previous_required_files | discovered_generation_files) - current_required_files
            for relative_name in obsolete_files:
                obsolete_path = checkpoint_dir / relative_name
                if obsolete_path.name.startswith(removable_prefixes):
                    obsolete_path.unlink(missing_ok=True)
    
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


def execute_paper_training(
    cfg,
    run_dir,
    avavla,
    action_head,
    dataloader,
    train_dataset,
    action_tokenizer,
    image_transform,
    source_config_path,
    avavla_config,
    distributed_state,
    device_id,
    num_patches,
    resume_state,
    val_dataloader=None,
) -> None:
    """Execute BC, latent warmup, true online PPO, then exit calibration."""
    log_step = int(resume_state.get("log_step", 0)) if cfg.resume else 0
    if cfg.resume:
        # A process may have logged unsaved work after the last atomic
        # checkpoint. Remove that tail before appending so global_step remains
        # unique and continuous after recovery.
        distributed_barrier()
        if distributed_state.is_main_process:
            metrics_path = run_dir / "metrics.jsonl"
            if metrics_path.exists():
                kept_rows = []
                with metrics_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if int(row.get("global_step", -1)) <= log_step:
                            kept_rows.append(json.dumps(row, sort_keys=True))
                temporary_path = metrics_path.with_name(f".{metrics_path.name}.tmp-{os.getpid()}")
                temporary_path.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")
                os.replace(temporary_path, metrics_path)
        distributed_barrier()
    stage_state_path = run_dir / "paper_stage_state.json"
    checkpoint_paper_state = resume_state.get("paper_state") if cfg.resume else None
    if checkpoint_paper_state:
        # This state was serialized in the same torch.save call as the model and
        # optimizer, so its counters cannot run ahead of the restored weights.
        saved_stage_state = dict(checkpoint_paper_state)
    elif cfg.resume and stage_state_path.exists():
        with stage_state_path.open("r", encoding="utf-8") as stream:
            candidate_stage_state = json.load(stream)
        if int(candidate_stage_state.get("log_step", -1)) <= log_step:
            saved_stage_state = candidate_stage_state
        else:
            saved_stage_state = {"stage": "bc", "log_step": log_step}
    else:
        saved_stage_state = {"stage": "bc", "log_step": log_step}
    resume_stage = saved_stage_state.get("stage", "bc")

    stage_order = {
        "bc": 0,
        "bc_complete": 1,
        "latent_warmup": 1,
        "latent_warmup_complete": 2,
        "online_ppo": 2,
        "online_ppo_complete": 3,
        "exit_calibration": 3,
        "complete": 4,
    }
    if resume_stage not in stage_order:
        raise ValueError(f"Unknown resume stage: {resume_stage}")
    resume_rank = stage_order[resume_stage]

    if resume_stage == "complete":
        distributed_barrier()
        if distributed_state.is_main_process:
            complete_payload = {
                "dataset": cfg.dataset_name,
                "log_step": log_step,
                "ppo_environment_steps": int(
                    saved_stage_state.get("global_env_steps", cfg.ppo_environment_steps)
                ),
                "exit_threshold": cfg.exit_threshold,
                "checkpoint_dir": str(run_dir),
            }
            (run_dir / "TRAINING_COMPLETE").write_text(
                json.dumps(complete_payload, indent=2) + "\n",
                encoding="utf-8",
            )
        distributed_barrier()
        return

    def restore_optimizer_for_stage(stage: str, optimizer, scheduler) -> None:
        if not cfg.resume or resume_stage != stage:
            return
        if "optimizer" not in resume_state or "scheduler" not in resume_state:
            return
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device_id)
        print(f"Restored {stage} optimizer/scheduler at global step {log_step}")

    def last_logged_metrics(stage: str, max_global_step: Optional[int] = None) -> Dict[str, float]:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return {}
        latest = {}
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("stage") == stage and (
                    max_global_step is None or int(row.get("global_step", -1)) <= max_global_step
                ):
                    latest = row.get("metrics", {})
        return latest
    data_iterator = iter(dataloader)

    def next_demo_batch():
        nonlocal data_iterator
        try:
            return next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            return next(data_iterator)

    def log_metrics(stage: str, metrics: Dict[str, float], grad_norm=None) -> None:
        numeric_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_metrics[key] = float(value)

        if grad_norm is None and "ppo_grad_norm" in numeric_metrics:
            grad_norm = numeric_metrics["ppo_grad_norm"]
        if grad_norm is not None:
            if torch.is_tensor(grad_norm):
                if grad_norm.numel() != 1:
                    raise ValueError("grad_norm must be scalar")
                grad_norm = grad_norm.detach().float().cpu().item()
            grad_norm = float(grad_norm)

        if dist.is_available() and dist.is_initialized() and numeric_metrics:
            metric_names = sorted(numeric_metrics)
            values = torch.tensor(
                [numeric_metrics[name] for name in metric_names],
                device=device_id,
                dtype=torch.float64,
            )
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values.div_(dist.get_world_size())
            numeric_metrics = {
                name: float(value)
                for name, value in zip(metric_names, values.cpu().tolist())
            }

        if dist.is_available() and dist.is_initialized() and grad_norm is not None:
            grad_value = torch.tensor(
                grad_norm,
                device=device_id,
                dtype=torch.float64,
            )
            dist.all_reduce(grad_value, op=dist.ReduceOp.SUM)
            grad_value.div_(dist.get_world_size())
            grad_norm = float(grad_value.cpu())
        if not distributed_state.is_main_process:
            return

        metric_row = {
            "timestamp_unix": time.time(),
            "stage": stage,
            "global_step": int(log_step),
            "metrics": numeric_metrics,
        }
        if grad_norm is not None:
            metric_row["grad_norm"] = float(grad_norm)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metric_row, sort_keys=True) + "\n")

        if log_step % cfg.wandb_log_freq:
            return
        printable = {key: round(value, 6) for key, value in numeric_metrics.items()}
        if "grad_norm" in metric_row:
            printable["grad_norm"] = round(metric_row["grad_norm"], 6)
        print(f"[{stage}] step={log_step} metrics={printable}", flush=True)
        if cfg.use_wandb:
            row = {f"{stage}/{key}": value for key, value in numeric_metrics.items()}
            if "grad_norm" in metric_row:
                row[f"{stage}/grad_norm"] = metric_row["grad_norm"]
            wandb.log(row, step=log_step)

    def maybe_validate(stage: str) -> None:
        if val_dataloader is None or cfg.val_freq <= 0 or log_step % cfg.val_freq:
            return
        validation_metrics = run_validation(
            avavla,
            action_head,
            val_dataloader,
            action_tokenizer,
            device_id,
            cfg,
            num_patches,
        )
        log_metrics(
            f"{stage}_validation",
            {f"val_{name}": value for name, value in validation_metrics.items()},
        )

    def save_checkpoint(
        stage: str,
        optimizer,
        scheduler,
        force: bool = False,
        **stage_metadata,
    ) -> None:
        if not force and (log_step == 0 or log_step % cfg.save_freq):
            return
        paper_state = {
            "stage": stage,
            "log_step": int(log_step),
            **stage_metadata,
        }
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
            paper_state=paper_state,
        )
        checkpoint_dir = (
            run_dir
            if cfg.save_latest_checkpoint_only
            else Path(str(run_dir) + f"--{log_step}_chkpt")
        )
        if distributed_state.is_main_process:
            _atomic_write_json(
                {**paper_state, "checkpoint_dir": str(checkpoint_dir)},
                run_dir / "paper_stage_state.json",
            )

    with tqdm.tqdm(
        total=cfg.max_steps,
        initial=min(log_step, cfg.max_steps),
        leave=False,
        disable=not distributed_state.is_main_process,
    ) as progress:
        # Stage 1: behavior cloning. Latent reasoning is deliberately bypassed.
        optimizer, scheduler, trainable_params = build_stage_optimizer(avavla, action_head, "bc", cfg)
        restore_optimizer_for_stage("bc", optimizer, scheduler)
        avavla.train()
        if action_head is not None:
            action_head.train()
        bc_start = (
            int(saved_stage_state.get("stage_step", min(log_step, cfg.bc_steps)))
            if resume_stage == "bc"
            else cfg.bc_steps
        )
        bc_start = max(0, min(cfg.bc_steps, bc_start))
        for stage_step in range(bc_start, cfg.bc_steps):
            optimizer.zero_grad(set_to_none=True)
            metric_rows = []
            for _ in range(cfg.grad_accumulation_steps):
                loss, metrics, _ = run_forward_pass_with_latent_reasoning(
                    avavla=avavla,
                    action_head=action_head,
                    batch=next_demo_batch(),
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=num_patches,
                    training=True,
                    include_rl_loss=False,
                    use_latent_state=False,
                )
                (loss / cfg.grad_accumulation_steps).backward()
                metric_rows.append(metrics)
            synchronize_gradients(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            log_step += 1
            progress.update()
            metrics = average_metric_dicts(metric_rows)
            metrics["stage_step"] = float(stage_step + 1)
            log_metrics("bc", metrics, grad_norm)
            maybe_validate("bc")
            save_checkpoint("bc", optimizer, scheduler, stage_step=stage_step + 1)
        if bc_start < cfg.bc_steps:
            save_checkpoint(
                "bc_complete",
                optimizer,
                scheduler,
                force=True,
                stage_step=cfg.bc_steps,
            )

        # Stage 2: action policy frozen; only smooth latent dynamics are warmed up.
        optimizer, scheduler, trainable_params = build_stage_optimizer(
            avavla, action_head, "latent_warmup", cfg
        )
        restore_optimizer_for_stage("latent_warmup", optimizer, scheduler)
        avavla.train()
        if action_head is not None:
            action_head.eval()
        warmup_start = (
            int(
                saved_stage_state.get(
                    "stage_step",
                    max(0, min(cfg.latent_warmup_steps, log_step - cfg.bc_steps)),
                )
            )
            if resume_stage == "latent_warmup"
            else (cfg.latent_warmup_steps if resume_rank > 1 else 0)
        )
        warmup_start = max(0, min(cfg.latent_warmup_steps, warmup_start))
        for stage_step in range(warmup_start, cfg.latent_warmup_steps):
            batch = next_demo_batch()
            input_ids = batch["input_ids"].to(device_id)
            attention_mask = batch["attention_mask"].to(device_id)
            labels = batch["labels"].to(device_id)
            pixel_values = move_to_device(batch["pixel_values"], device_id)
            wrist = batch.get("pixel_values_wrist")
            if wrist is not None:
                wrist = move_to_device(wrist, device_id)
            proprio = batch.get("proprio")
            if proprio is not None:
                proprio = proprio.to(device_id).to(torch.bfloat16)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                history_states = get_batch_history_states(
                    avavla,
                    batch,
                    device_id,
                    input_ids,
                    attention_mask,
                    labels,
                )
                loss, metrics, _ = avavla(
                    training_objective="latent_warmup",
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=cast_floating_tensors(pixel_values, torch.bfloat16),
                    pixel_values_wrist=(
                        cast_floating_tensors(wrist, torch.bfloat16) if wrist is not None else None
                    ),
                    proprio=proprio,
                    labels=labels,
                    history_states=history_states,
                    objective_kwargs={"num_steps": cfg.max_reasoning_steps},
                )
            loss.backward()
            synchronize_gradients(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            log_step += 1
            progress.update()
            metrics["stage_step"] = float(stage_step + 1)
            log_metrics("latent_warmup", metrics, grad_norm)
            maybe_validate("latent_warmup")
            save_checkpoint(
                "latent_warmup",
                optimizer,
                scheduler,
                stage_step=stage_step + 1,
            )
        if warmup_start < cfg.latent_warmup_steps:
            save_checkpoint(
                "latent_warmup_complete",
                optimizer,
                scheduler,
                force=True,
                stage_step=cfg.latent_warmup_steps,
            )

        # Stage 3: stochastic latent updates receive only true environment task success reward.
        if not cfg.online_ppo:
            raise ValueError("Paper reproduction requires online_ppo=True.")
        from vla_scripts.online_policy import collect_online_rollout

        is_calvin = cfg.dataset_name == "calvin_abc"
        if is_calvin:
            from experiments.robot.calvin.online_rollout import CalvinVectorRollout
        else:
            from experiments.robot.libero.online_rollout import LiberoVectorRollout

            task_suite_names = LiberoVectorRollout.suites_for_dataset(cfg.dataset_name)
            allowed_suites = {name.strip() for name in cfg.libero_task_suites.split(",") if name.strip()}
            disabled_suites = [name for name in task_suite_names if name not in allowed_suites]
            if disabled_suites:
                raise ValueError(
                    f"Suites {disabled_suites} are not enabled by "
                    f"libero_task_suites={cfg.libero_task_suites!r}."
                )
        optimizer, scheduler, trainable_params = build_stage_optimizer(avavla, action_head, "ppo", cfg)
        restore_optimizer_for_stage("online_ppo", optimizer, scheduler)
        online_resume_metrics = last_logged_metrics("online_ppo", max_global_step=log_step)
        if resume_stage == "online_ppo":
            global_env_steps = int(
                saved_stage_state.get(
                    "global_env_steps",
                    online_resume_metrics.get("global_env_steps", 0),
                )
            )
            ppo_update = int(
                saved_stage_state.get(
                    "ppo_update",
                    online_resume_metrics.get("ppo_update", 0),
                )
            )
        elif resume_rank > 2:
            global_env_steps = int(
                saved_stage_state.get(
                    "global_env_steps",
                    online_resume_metrics.get("global_env_steps", cfg.ppo_environment_steps),
                )
            )
            ppo_update = int(
                saved_stage_state.get(
                    "ppo_update",
                    online_resume_metrics.get("ppo_update", cfg.ppo_optimizer_steps or 0),
                )
            )
        else:
            global_env_steps = 0
            ppo_update = 0

        last_rollout = None
        calibration_rollouts = deque(maxlen=max(1, cfg.exit_calibration_buffer_rollouts))
        calibration_buffer_path = run_dir / (
            f"exit_calibration_buffer_rank{distributed_state.process_index}.pt"
        )
        if cfg.resume and calibration_buffer_path.exists():
            saved_rollouts = torch.load(calibration_buffer_path, map_location=device_id)
            calibration_rollouts.extend(saved_rollouts)
            if calibration_rollouts:
                last_rollout = calibration_rollouts[-1]

        def save_calibration_buffer() -> None:
            if not calibration_rollouts:
                return
            temporary_path = calibration_buffer_path.with_suffix(".pt.tmp")
            torch.save(list(calibration_rollouts), temporary_path)
            os.replace(temporary_path, calibration_buffer_path)

        if resume_rank <= 2:
            if is_calvin:
                collector = CalvinVectorRollout(
                    dataset_root=cfg.data_root_dir,
                    num_envs=cfg.online_envs_per_rank,
                    rank=distributed_state.process_index,
                    world_size=distributed_state.num_processes,
                    seed=cfg.seed,
                )
            else:
                collector = LiberoVectorRollout(
                    task_suite_name=task_suite_names,
                    num_envs=cfg.online_envs_per_rank,
                    rank=distributed_state.process_index,
                    world_size=distributed_state.num_processes,
                    seed=cfg.seed,
                    num_steps_wait=cfg.online_num_steps_wait,
                    env_img_res=cfg.online_env_img_res,
                )
            try:
                while global_env_steps < cfg.ppo_environment_steps:
                    avavla.eval()
                    if action_head is not None:
                        action_head.eval()
                    rollout, local_env_steps, online_metrics = collect_online_rollout(
                        avavla=avavla,
                        action_head=action_head,
                        collector=collector,
                        image_transform=image_transform,
                        # Every online observation carries its dataset-specific statistics key.
                        # This supports the four-suite LIBERO mixture and native CALVIN alike.
                        unnorm_key=None,
                        rollout_size=cfg.ppo_rollout_size_per_rank,
                        center_crop=cfg.online_center_crop,
                        action_policy_std=cfg.action_ppo_std,
                    )
                    if cfg.require_online_task_rewards and "ppo_rewards" not in rollout:
                        raise RuntimeError("Online PPO rollout is missing true task rewards.")

                    online_metrics = reduce_online_rollout_metrics(online_metrics, device_id)
                    step_tensor = torch.tensor(local_env_steps, device=device_id, dtype=torch.long)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
                    global_env_steps += int(step_tensor.item())

                    target_metrics = prepare_temporal_ppo_targets(avavla, rollout, cfg)
                    avavla.train()
                    ppo_metrics = run_ppo_updates(
                        avavla,
                        action_head,
                        optimizer,
                        scheduler,
                        trainable_params,
                        cfg,
                        rollout,
                    )
                    if cfg.ppo_bc_aux_coef > 0:
                        avavla.train()
                        if action_head is not None:
                            action_head.train()
                        optimizer.zero_grad(set_to_none=True)
                        bc_aux_loss, bc_aux_metrics, _ = run_forward_pass_with_latent_reasoning(
                            avavla=avavla,
                            action_head=action_head,
                            batch=next_demo_batch(),
                            action_tokenizer=action_tokenizer,
                            device_id=device_id,
                            cfg=cfg,
                            num_patches=num_patches,
                            training=True,
                            include_rl_loss=False,
                            use_latent_state=True,
                        )
                        (cfg.ppo_bc_aux_coef * bc_aux_loss).backward()
                        synchronize_gradients(trainable_params)
                        bc_aux_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        ppo_metrics.update(
                            {
                                f"ppo_bc_{name}": value
                                for name, value in bc_aux_metrics.items()
                            }
                        )
                        ppo_metrics["ppo_bc_grad_norm"] = float(
                            bc_aux_grad_norm.detach().cpu()
                            if torch.is_tensor(bc_aux_grad_norm)
                            else bc_aux_grad_norm
                        )
                        ppo_metrics["ppo_joint_bc_update"] = 1.0
                    ppo_update += 1
                    log_step += 1
                    progress.update()
                    ppo_metrics.update(online_metrics)
                    ppo_metrics.update(target_metrics)
                    ppo_metrics["global_env_steps"] = float(global_env_steps)
                    ppo_metrics["ppo_update"] = float(ppo_update)
                    log_metrics("online_ppo", ppo_metrics, ppo_metrics.get("ppo_grad_norm"))
                    maybe_validate("online_ppo")
                    # Stage 4 only needs latent states, masks, and batch size.
                    # Drop replay pixels/tokens here so the 128-rollout
                    # calibration buffer does not retain the full OFT inputs.
                    calibration_rollout = {
                        name: rollout[name].detach()
                        for name in ("ppo_rewards", "next_latent_states", "valid_mask")
                    }
                    last_rollout = calibration_rollout
                    calibration_rollouts.append(calibration_rollout)
                    ppo_checkpoint_due = should_checkpoint_ppo(
                        cfg,
                        ppo_update=ppo_update,
                        global_env_steps=global_env_steps,
                    )
                    if ppo_checkpoint_due:
                        # Persist rollout data before any checkpoint whose stage
                        # counters depend on it, preventing an unrecoverable
                        # exit-calibration resume point.
                        save_calibration_buffer()
                    save_checkpoint(
                        "online_ppo",
                        optimizer,
                        scheduler,
                        force=ppo_checkpoint_due,
                        global_env_steps=global_env_steps,
                        ppo_update=ppo_update,
                    )
                save_calibration_buffer()
            finally:
                collector.close()
            if last_rollout is None:
                raise RuntimeError("Online PPO completed without a recoverable rollout buffer.")
            save_checkpoint(
                "online_ppo_complete",
                optimizer,
                scheduler,
                force=True,
                global_env_steps=global_env_steps,
                ppo_update=ppo_update,
            )

        if not calibration_rollouts and resume_rank <= 3:
            raise RuntimeError(
                f"Missing exit-calibration rollout buffer at {calibration_buffer_path}; "
                "resume the online_ppo checkpoint that precedes exit calibration."
            )

        # Stage 4: freeze everything except g_omega and calibrate from critic value of computation.
        optimizer, scheduler, trainable_params = build_stage_optimizer(
            avavla, action_head, "exit_calibration", cfg
        )
        restore_optimizer_for_stage("exit_calibration", optimizer, scheduler)
        exit_start = (
            int(saved_stage_state.get("stage_step", 0))
            if resume_stage == "exit_calibration"
            else (cfg.exit_calibration_steps if resume_rank > 3 else 0)
        )
        exit_start = max(0, min(cfg.exit_calibration_steps, exit_start))
        for stage_step in range(exit_start, cfg.exit_calibration_steps):
            calibration_rollout = calibration_rollouts[
                random.randrange(len(calibration_rollouts))
            ]
            local_calibration_batch = max(
                1,
                min(
                    calibration_rollout["ppo_rewards"].shape[0],
                    cfg.ppo_minibatch_size // max(1, distributed_state.num_processes),
                ),
            )
            indices = torch.randperm(
                calibration_rollout["ppo_rewards"].shape[0],
                device=calibration_rollout["ppo_rewards"].device,
            )[:local_calibration_batch]
            minibatch = slice_ppo_rollout(calibration_rollout, indices)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = avavla(
                training_objective="exit_calibration",
                reasoning_trajectories=minibatch,
                objective_kwargs={
                    "lookahead": cfg.exit_calibration_lookahead,
                    "delta": cfg.exit_calibration_delta,
                },
            )
            loss.backward()
            synchronize_gradients(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            log_step += 1
            progress.update()
            metrics["stage_step"] = float(stage_step + 1)
            log_metrics("exit_calibration", metrics, grad_norm)
            save_checkpoint(
                "exit_calibration",
                optimizer,
                scheduler,
                force=(stage_step + 1) % cfg.exit_checkpoint_interval_steps == 0,
                stage_step=stage_step + 1,
                global_env_steps=global_env_steps,
                ppo_update=ppo_update,
            )
        save_checkpoint(
            "complete",
            optimizer,
            scheduler,
            force=True,
            stage_step=cfg.exit_calibration_steps,
            global_env_steps=global_env_steps,
            ppo_update=ppo_update,
        )

    distributed_barrier()
    if distributed_state.is_main_process:
        _atomic_write_json(
            {
                "dataset": cfg.dataset_name,
                "log_step": log_step,
                "ppo_environment_steps": global_env_steps,
                "exit_threshold": cfg.exit_threshold,
                "checkpoint_dir": str(
                    run_dir
                    if cfg.save_latest_checkpoint_only
                    else Path(str(run_dir) + f"--{log_step}_chkpt")
                ),
            },
            run_dir / "TRAINING_COMPLETE",
        )
    distributed_barrier()


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
    rank_seed = int(cfg.seed) + int(distributed_state.process_index)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)

    resolve_paper_training_budget(cfg, distributed_state.num_processes)
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    if distributed_state.is_main_process and cfg.use_wandb:
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
    if distributed_state.is_main_process:
        _link_or_copy_base_checkpoint(
            source_checkpoint_path,
            run_dir / "checkpoints" / "latest-checkpoint.pt",
        )
    distributed_barrier()
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
                hidden_dim=cfg.action_hidden_dim,
                action_dim=ACTION_DIM
            ).to(torch.bfloat16).to(device_id),
            device_id
        )
    else:
        action_head = None
    
    # Get number of patches
    num_patches = avavla.module.vision_backbone.num_patches
    
    optimizer, scheduler, trainable_params = build_stage_optimizer(avavla, action_head, "bc", cfg)
    print(f"# BC trainable params: {sum(p.numel() for p in trainable_params)}")
    resume_state = load_resume_state(cfg, avavla, action_head, None, None, device_id)
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=cfg.use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    
    if cfg.dataset_name == "calvin_abc":
        from experiments.robot.calvin.dataset import CalvinDiskDataset

        train_dataset = CalvinDiskDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            window_size=cfg.history_window_size,
            train=True,
            seed=cfg.seed,
        )
    else:
        train_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(avavla.module.vision_backbone.default_image_resolution[-2:]),
            window_size=cfg.history_window_size,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
            seed=cfg.seed,
        )
    avavla.module.norm_stats = train_dataset.dataset_statistics
    
    if distributed_state.is_main_process:
        if not (run_dir / "dataset_statistics.json").exists():
            save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        _copy_if_different(source_config_path, run_dir / "config.json")
        _atomic_write_json(avavla_config, run_dir / "avavla_config.json")
    
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
        if cfg.dataset_name == "calvin_abc":
            val_dataset = CalvinDiskDataset(
                cfg.data_root_dir,
                cfg.dataset_name,
                batch_transform,
                window_size=cfg.history_window_size,
                train=False,
                seed=cfg.seed,
            )
        else:
            val_dataset = RLDSDataset(
                cfg.data_root_dir,
                cfg.dataset_name,
                batch_transform,
                resize_resolution=tuple(avavla.module.vision_backbone.default_image_resolution[-2:]),
                window_size=cfg.history_window_size,
                shuffle_buffer_size=cfg.shuffle_buffer_size,
                train=False,
                image_aug=False,
                seed=cfg.seed,
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

    execute_paper_training(
        cfg=cfg,
        run_dir=run_dir,
        avavla=avavla,
        action_head=action_head,
        dataloader=dataloader,
        train_dataset=train_dataset,
        action_tokenizer=action_tokenizer,
        image_transform=image_transform,
        source_config_path=source_config_path,
        avavla_config=avavla_config,
        distributed_state=distributed_state,
        device_id=device_id,
        num_patches=num_patches,
        resume_state=resume_state,
        val_dataloader=val_dataloader,
    )
    return


if __name__ == "__main__":
    finetune_avavla()
