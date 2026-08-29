"""
Train and evaluate AVA-VLA on a compact LIBERO RLDS sample.

The default run uses exactly `--train-samples` step samples and executes the production training sequence:
behavior cloning, latent reasoning warmup, PPO joint RL fine-tuning, and exit-gate calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prismatic.models.vlas.avavla import ExitGate, LatentTransition, ReasoningPolicy, ValueFunction


try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass


@dataclass
class AVAVLALiberoTrainConfig:
    tfrecord: str
    output_dir: str
    samples_json: Optional[str] = None
    train_samples: int = 10
    eval_samples: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    policy_lr: float = 3e-5
    critic_lr: float = 1e-4
    latent_dim: int = 128
    obs_dim: int = 128
    hidden_dim: int = 256
    update_dim: int = 32
    reasoning_policy_type: str = "gaussian"
    reasoning_steps: int = 5
    exit_threshold: float = 0.55
    bc_steps: int = 10
    latent_warmup_steps: int = 5
    ppo_steps: int = 10
    exit_calibration_steps: int = 5
    ppo_clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coef: float = 0.01
    smoothness_coef: float = 0.1
    value_coef: float = 0.5
    exit_loss_coef: float = 0.1
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 4
    exit_calibration_lookahead: int = 3
    exit_calibration_delta: float = 0.05
    exit_calibration_target_rate: Optional[float] = None
    seed: int = 7
    device: str = "cuda"


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("avavla_libero_10sample")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _decode_image_stats(encoded: bytes) -> np.ndarray:
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False).numpy().astype(np.float32) / 255.0
    if image.ndim != 3:
        image = np.reshape(image, (*image.shape[:2], 3))
    center = image[
        image.shape[0] // 4 : max(image.shape[0] // 4 + 1, image.shape[0] * 3 // 4),
        image.shape[1] // 4 : max(image.shape[1] // 4 + 1, image.shape[1] * 3 // 4),
    ]
    stats = np.concatenate(
        [
            image.mean(axis=(0, 1)),
            image.std(axis=(0, 1)),
            image.min(axis=(0, 1)),
            image.max(axis=(0, 1)),
            center.mean(axis=(0, 1)),
            center.std(axis=(0, 1)),
        ]
    )
    return stats.astype(np.float32)


def _text_features(text: str, dim: int = 32) -> np.ndarray:
    features = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        features[idx] += 1.0
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def _reshape_feature(values: List[float], step_count: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size % step_count != 0:
        raise ValueError(f"{name} has {arr.size} values, not divisible by {step_count} steps")
    return arr.reshape(step_count, arr.size // step_count)


def load_libero_samples(tfrecord: Path, limit: Optional[int] = 10) -> List[Dict[str, Any]]:
    if not tfrecord.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord}")

    samples: List[Dict[str, Any]] = []
    for raw in tf.data.TFRecordDataset([str(tfrecord)]):
        example = tf.train.Example.FromString(raw.numpy())
        features = example.features.feature

        actions_raw = list(features["steps/action"].float_list.value)
        image_raw = list(features["steps/observation/image"].bytes_list.value)
        wrist_raw = list(features["steps/observation/wrist_image"].bytes_list.value)
        languages = [x.decode("utf-8") for x in features["steps/language_instruction"].bytes_list.value]
        step_count = len(languages)
        if step_count == 0:
            continue

        actions = _reshape_feature(actions_raw, step_count, "steps/action")
        joint_state = _reshape_feature(list(features["steps/observation/joint_state"].float_list.value), step_count, "joint_state")
        state = _reshape_feature(list(features["steps/observation/state"].float_list.value), step_count, "state")
        rewards = np.asarray(list(features["steps/reward"].float_list.value), dtype=np.float32)

        for idx in range(step_count):
            obs_feature = np.concatenate(
                [
                    state[idx],
                    joint_state[idx],
                    _decode_image_stats(image_raw[idx]),
                    _decode_image_stats(wrist_raw[idx]),
                    _text_features(languages[idx]),
                ]
            ).astype(np.float32)
            samples.append(
                {
                    "episode_step": idx,
                    "language": languages[idx],
                    "observation": obs_feature,
                    "action": actions[idx].astype(np.float32),
                    "reward": float(rewards[idx]) if idx < rewards.shape[0] else 0.0,
                }
            )
            if limit is not None and len(samples) >= limit:
                return samples
    if not samples:
        raise RuntimeError(f"No samples could be loaded from {tfrecord}")
    return samples


def save_samples_json(samples: List[Dict[str, Any]], path: Path) -> None:
    serializable = []
    for sample in samples:
        serializable.append(
            {
                "episode_step": int(sample["episode_step"]),
                "language": sample["language"],
                "observation": sample["observation"].tolist(),
                "action": sample["action"].tolist(),
                "reward": float(sample["reward"]),
            }
        )
    path.write_text(json.dumps(serializable, indent=2))


def load_samples_json(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if limit is not None:
        data = data[:limit]
    return [
        {
            "episode_step": int(sample["episode_step"]),
            "language": sample["language"],
            "observation": np.asarray(sample["observation"], dtype=np.float32),
            "action": np.asarray(sample["action"], dtype=np.float32),
            "reward": float(sample.get("reward", 0.0)),
        }
        for sample in data
    ]


class LiberoStepDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "observation": torch.from_numpy(sample["observation"]),
            "action": torch.from_numpy(sample["action"]),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "language": sample["language"],
            "episode_step": sample["episode_step"],
        }


class FourLayerActionPolicy(nn.Module):
    """Continuous action head: 4-layer MLP with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, obs], dim=-1))


class LightweightAVAVLA(nn.Module):
    def __init__(self, input_dim: int, cfg: AVAVLALiberoTrainConfig, action_dim: int = 7) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.obs_dim),
            nn.LayerNorm(cfg.obs_dim),
        )
        self.initial_latent_proj = nn.Sequential(
            nn.LayerNorm(cfg.obs_dim),
            nn.Linear(cfg.obs_dim, cfg.latent_dim),
            nn.Tanh(),
        )
        self.reasoning_policy = ReasoningPolicy(
            latent_dim=cfg.latent_dim,
            obs_dim=cfg.obs_dim,
            hidden_dim=cfg.hidden_dim,
            update_dim=cfg.update_dim,
            num_heads=4,
            num_layers=4,
            dropout=0.1,
            policy_type=cfg.reasoning_policy_type,
        )
        self.latent_transition = LatentTransition(
            latent_dim=cfg.latent_dim,
            obs_dim=cfg.obs_dim,
            hidden_dim=cfg.hidden_dim,
            update_dim=cfg.update_dim,
            num_heads=4,
            dropout=0.1,
        )
        self.exit_gate = ExitGate(cfg.latent_dim, hidden_dim=cfg.hidden_dim, dropout=0.1)
        self.value_function = ValueFunction(cfg.latent_dim, hidden_dim=cfg.hidden_dim, dropout=0.1)
        self.action_policy = FourLayerActionPolicy(cfg.latent_dim + cfg.obs_dim, cfg.hidden_dim, action_dim)

    def set_stage(self, stage: str) -> None:
        for param in self.parameters():
            param.requires_grad = False

        trainable_modules: Iterable[nn.Module]
        if stage == "bc":
            trainable_modules = (self.obs_encoder, self.initial_latent_proj, self.action_policy)
        elif stage == "latent_warmup":
            trainable_modules = (self.reasoning_policy, self.latent_transition)
        elif stage == "ppo":
            trainable_modules = (
                self.obs_encoder,
                self.initial_latent_proj,
                self.reasoning_policy,
                self.latent_transition,
                self.value_function,
                self.action_policy,
            )
        elif stage == "exit_calibration":
            trainable_modules = (self.exit_gate,)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def configure_optimizer(self, cfg: AVAVLALiberoTrainConfig) -> torch.optim.Optimizer:
        groups = [
            {
                "params": [
                    p
                    for module in (self.obs_encoder, self.initial_latent_proj, self.latent_transition, self.action_policy)
                    for p in module.parameters()
                    if p.requires_grad
                ],
                "lr": cfg.learning_rate,
            },
            {"params": [p for p in self.reasoning_policy.parameters() if p.requires_grad], "lr": cfg.policy_lr},
            {"params": [p for p in self.value_function.parameters() if p.requires_grad], "lr": cfg.critic_lr},
            {"params": [p for p in self.exit_gate.parameters() if p.requires_grad], "lr": cfg.critic_lr},
        ]
        groups = [group for group in groups if group["params"]]
        if not groups:
            raise RuntimeError("No trainable parameters for the selected stage")
        return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

    def forward(
        self,
        observation: torch.Tensor,
        training: bool = True,
        adaptive_exit: bool = False,
        fixed_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        obs = self.obs_encoder(observation)
        z = self.initial_latent_proj(obs)
        max_steps = fixed_steps or self.cfg.reasoning_steps

        log_probs, old_log_probs, entropies = [], [], []
        exit_scores, values = [], []
        latent_states, next_latent_states, update_actions = [], [], []
        step_action_preds = []
        active = torch.ones(observation.shape[0], dtype=torch.bool, device=observation.device)
        steps_per_sample = torch.zeros(observation.shape[0], dtype=torch.long, device=observation.device)
        final_z = z

        for _ in range(max_steps):
            z_before = z
            policy_output = self.reasoning_policy(z_before, obs)
            update_action, log_prob, entropy = self.reasoning_policy.sample_update_action(
                policy_output,
                training=training,
            )
            z_next = self.latent_transition(z_before, obs, update_action)
            exit_score = self.exit_gate(z_next).squeeze(-1)

            latent_states.append(z_before)
            next_latent_states.append(z_next)
            update_actions.append(update_action)
            log_probs.append(log_prob)
            old_log_probs.append(log_prob.detach())
            entropies.append(entropy)
            exit_scores.append(exit_score)
            values.append(self.value_function(z_before).squeeze(-1))
            step_action_preds.append(self.action_policy(obs, z_next))

            if adaptive_exit and not training:
                z = torch.where(active.unsqueeze(-1), z_next, z)
                newly_finished = active & (exit_score > self.cfg.exit_threshold)
                final_z = torch.where(newly_finished.unsqueeze(-1), z, final_z)
                steps_per_sample += active.long()
                active = active & ~newly_finished
                if not active.any():
                    break
            else:
                z = z_next
                steps_per_sample += 1

        if adaptive_exit and not training:
            final_z = torch.where(active.unsqueeze(-1), z, final_z)
        else:
            final_z = z

        pred_action = self.action_policy(obs, final_z)
        return {
            "pred_action": pred_action,
            "step_action_preds": torch.stack(step_action_preds, dim=1),
            "log_probs": torch.stack(log_probs, dim=1),
            "old_log_probs": torch.stack(old_log_probs, dim=1),
            "entropies": torch.stack(entropies, dim=1),
            "exit_scores": torch.stack(exit_scores, dim=1),
            "values": torch.stack(values, dim=1),
            "latent_states": torch.stack(latent_states, dim=1),
            "next_latent_states": torch.stack(next_latent_states, dim=1),
            "update_actions": torch.stack(update_actions, dim=1),
            "obs_encoding": obs,
            "final_latent": final_z,
            "steps_per_sample": steps_per_sample,
        }

    def evaluate_rollout(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = rollout["obs_encoding"]
        latent_states = rollout["latent_states"]
        next_latent_states = rollout["next_latent_states"]
        update_actions = rollout["update_actions"]
        batch_size, num_steps, _ = latent_states.shape

        obs_steps = obs.unsqueeze(1).expand(-1, num_steps, -1)
        policy_output = self.reasoning_policy(
            latent_states.reshape(batch_size * num_steps, -1),
            obs_steps.reshape(batch_size * num_steps, -1),
        )
        log_probs, entropies = self.reasoning_policy.evaluate_update_action(
            policy_output,
            update_actions.reshape(batch_size * num_steps, -1),
        )
        values = self.value_function(latent_states).squeeze(-1)
        exit_scores = self.exit_gate(next_latent_states.reshape(batch_size * num_steps, -1)).reshape(
            batch_size,
            num_steps,
        )
        pred_action = self.action_policy(obs, rollout["final_latent"])

        return {
            "pred_action": pred_action,
            "log_probs": log_probs.reshape(batch_size, num_steps),
            "old_log_probs": rollout["old_log_probs"],
            "entropies": entropies.reshape(batch_size, num_steps),
            "exit_scores": exit_scores,
            "values": values,
            "latent_states": latent_states,
            "next_latent_states": next_latent_states,
            "update_actions": update_actions,
            "ppo_rewards": rollout.get("ppo_rewards"),
        }


def _proxy_reward(pred: torch.Tensor, target: torch.Tensor, env_reward: torch.Tensor) -> torch.Tensor:
    action_error = torch.mean(torch.abs(pred.detach() - target), dim=-1)
    action_proxy = torch.exp(-action_error).clamp(0.0, 1.0)
    return torch.maximum(action_proxy, env_reward.clamp(0.0, 1.0))


def behavior_cloning_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
    action_loss = F.l1_loss(outputs["pred_action"], target)
    return {
        "total_loss": action_loss,
        "action_loss": action_loss,
        "mean_abs_error": torch.mean(torch.abs(outputs["pred_action"].detach() - target)),
    }


def latent_warmup_loss(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    smoothness = (outputs["next_latent_states"] - outputs["latent_states"]).pow(2).mean()
    zero_mean = outputs["update_actions"].pow(2).mean()
    total = smoothness + 0.001 * zero_mean
    return {
        "total_loss": total,
        "smoothness_loss": smoothness,
        "update_energy": zero_mean,
    }


def ppo_joint_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    env_reward: torch.Tensor,
    cfg: AVAVLALiberoTrainConfig,
) -> Dict[str, torch.Tensor]:
    action_loss = F.l1_loss(outputs["pred_action"], target)
    reward_target = outputs.get("ppo_rewards")
    if reward_target is None:
        reward_target = _proxy_reward(outputs["pred_action"], target, env_reward)
    reward_target = reward_target.to(device=target.device, dtype=target.dtype)

    log_probs = outputs["log_probs"]
    old_log_probs = outputs["old_log_probs"]
    entropies = outputs["entropies"]
    values = outputs["values"]
    valid_mask = torch.ones_like(log_probs)
    smoothness = (outputs["next_latent_states"] - outputs["latent_states"]).pow(2).mean(dim=-1)
    immediate_rewards = (
        reward_target.unsqueeze(1).expand_as(log_probs)
        - cfg.entropy_coef * entropies.detach()
        - cfg.smoothness_coef * smoothness.detach()
    )

    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    deltas = immediate_rewards + cfg.gamma * next_values.detach() - values.detach()
    advantages = torch.zeros_like(values)
    running_advantage = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    for step in reversed(range(values.shape[1])):
        running_advantage = deltas[:, step] + cfg.gamma * cfg.gae_lambda * running_advantage
        advantages[:, step] = running_advantage
    returns = advantages + values.detach()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

    ratio = torch.exp((log_probs - old_log_probs).clamp(-20.0, 20.0))
    unclipped = ratio * advantages.detach()
    clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip_ratio, 1.0 + cfg.ppo_clip_ratio) * advantages.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns.detach())
    exit_targets = reward_target.detach().unsqueeze(1).expand_as(outputs["exit_scores"])
    exit_loss = F.binary_cross_entropy(outputs["exit_scores"].clamp(1e-6, 1.0 - 1e-6), exit_targets)
    total = action_loss + policy_loss + cfg.value_coef * value_loss + cfg.exit_loss_coef * exit_loss

    return {
        "total_loss": total,
        "action_loss": action_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "exit_loss": exit_loss,
        "mean_reward": reward_target.mean(),
        "mean_abs_error": torch.mean(torch.abs(outputs["pred_action"].detach() - target)),
        "mean_exit_score": outputs["exit_scores"][:, -1].mean(),
        "ppo_ratio_mean": ratio.mean(),
        "ppo_clip_fraction": ((ratio - 1.0).abs() > cfg.ppo_clip_ratio).to(dtype=ratio.dtype).mean(),
        "smoothness_loss": smoothness.mean(),
    }


def exit_calibration_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    cfg: AVAVLALiberoTrainConfig,
) -> Dict[str, torch.Tensor]:
    step_errors = torch.mean(torch.abs(outputs["step_action_preds"].detach() - target.unsqueeze(1)), dim=-1)
    labels = torch.zeros_like(step_errors)
    for step in range(step_errors.shape[1]):
        future_step = min(step + cfg.exit_calibration_lookahead, step_errors.shape[1] - 1)
        improvement = step_errors[:, step] - step_errors[:, future_step]
        labels[:, step] = (improvement < cfg.exit_calibration_delta).to(dtype=labels.dtype)

    exit_loss = F.binary_cross_entropy(outputs["exit_scores"].clamp(1e-6, 1.0 - 1e-6), labels)
    return {
        "total_loss": exit_loss,
        "exit_loss": exit_loss,
        "exit_positive_rate": labels.mean(),
        "mean_exit_score": outputs["exit_scores"].mean(),
    }


def next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def detach_rollout(outputs: Dict[str, torch.Tensor], ppo_rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
    rollout = {
        key: value.detach()
        for key, value in outputs.items()
        if torch.is_tensor(value)
    }
    rollout["old_log_probs"] = outputs["log_probs"].detach()
    rollout["ppo_rewards"] = ppo_rewards.detach()
    return rollout


def slice_rollout(rollout: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch_size = rollout["ppo_rewards"].shape[0]
    sliced = {}
    for key, value in rollout.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            sliced[key] = value.index_select(0, indices)
        else:
            sliced[key] = value
    return sliced


def average_loss_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def train_stage(
    model: LightweightAVAVLA,
    dataloader: DataLoader,
    stage: str,
    steps: int,
    cfg: AVAVLALiberoTrainConfig,
    device: torch.device,
    logger: logging.Logger,
    metrics_path: Path,
    global_step: int,
) -> int:
    if steps <= 0:
        return global_step
    model.set_stage(stage)
    optimizer = model.configure_optimizer(cfg)
    iterator = iter(dataloader)
    model.train()

    for local_step in range(1, steps + 1):
        batch, iterator = next_batch(iterator, dataloader)
        observation = batch["observation"].to(device=device, dtype=torch.float32)
        target = batch["action"].to(device=device, dtype=torch.float32)
        reward = batch["reward"].to(device=device, dtype=torch.float32)

        latest_row = {}
        if stage == "ppo":
            with torch.no_grad():
                rollout_outputs = model(observation, training=True, adaptive_exit=False)
                rollout_rewards = _proxy_reward(rollout_outputs["pred_action"], target, reward)
                rollout = detach_rollout(rollout_outputs, rollout_rewards)

            ppo_rows: List[Dict[str, float]] = []
            minibatch_size = max(1, min(cfg.ppo_minibatch_size, observation.shape[0]))
            for epoch in range(max(1, cfg.ppo_epochs)):
                permutation = torch.randperm(observation.shape[0], device=device)
                for start in range(0, observation.shape[0], minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    outputs = model.evaluate_rollout(slice_rollout(rollout, indices))
                    losses = ppo_joint_loss(outputs, target.index_select(0, indices), reward.index_select(0, indices), cfg)

                    optimizer.zero_grad(set_to_none=True)
                    losses["total_loss"].backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        1.0,
                    )
                    optimizer.step()
                    row = {
                        "global_step": global_step + 1,
                        "stage": stage,
                        "stage_step": local_step,
                        "ppo_epoch": epoch + 1,
                        "ppo_minibatch_size": int(indices.numel()),
                        "grad_norm": float(grad_norm),
                        **{name: float(value.detach().cpu()) for name, value in losses.items()},
                    }
                    ppo_rows.append(row)
                    global_step += 1
            latest_row = average_loss_rows(ppo_rows)
            latest_row.update(
                {
                    "global_step": global_step,
                    "stage": stage,
                    "stage_step": local_step,
                    "ppo_epoch": cfg.ppo_epochs,
                    "ppo_minibatch_size": minibatch_size,
                }
            )
        else:
            outputs = model(observation, training=True, adaptive_exit=False)
            if stage == "bc":
                losses = behavior_cloning_loss(outputs, target)
            elif stage == "latent_warmup":
                losses = latent_warmup_loss(outputs)
            elif stage == "exit_calibration":
                losses = exit_calibration_loss(outputs, target, cfg)
            else:
                raise ValueError(stage)

            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()
            latest_row = {
                "global_step": global_step + 1,
                "stage": stage,
                "stage_step": local_step,
                "grad_norm": float(grad_norm),
                **{name: float(value.detach().cpu()) for name, value in losses.items()},
            }
            global_step += 1

        with metrics_path.open("a") as f:
            f.write(json.dumps(latest_row) + "\n")
        logger.info(
            "stage=%s step=%02d/%02d total=%.6f%s",
            stage,
            local_step,
            steps,
            latest_row["total_loss"],
            f" mae={latest_row['mean_abs_error']:.6f}" if "mean_abs_error" in latest_row else "",
        )
    return global_step


def evaluate(model: LightweightAVAVLA, samples: List[Dict[str, Any]], device: torch.device, count: int) -> Dict[str, Any]:
    model.eval()
    rows = []
    with torch.no_grad():
        for index, sample in enumerate(samples[:count]):
            observation = torch.from_numpy(sample["observation"]).unsqueeze(0).to(device)
            target = torch.from_numpy(sample["action"]).unsqueeze(0).to(device)
            outputs = model(observation, training=False, adaptive_exit=True)
            pred = outputs["pred_action"]
            abs_error = torch.abs(pred - target)
            rows.append(
                {
                    "sample_index": index,
                    "episode_step": int(sample["episode_step"]),
                    "language": sample["language"],
                    "target_action": target.squeeze(0).cpu().tolist(),
                    "predicted_action": pred.squeeze(0).cpu().tolist(),
                    "mean_abs_error": float(abs_error.mean().cpu()),
                    "l2_error": float(torch.linalg.norm(pred - target).cpu()),
                    "exit_scores": outputs["exit_scores"].squeeze(0).cpu().tolist(),
                    "reasoning_steps": int(outputs["steps_per_sample"].item()),
                }
            )

    mean_abs_error = float(np.mean([row["mean_abs_error"] for row in rows])) if rows else math.nan
    mean_l2_error = float(np.mean([row["l2_error"] for row in rows])) if rows else math.nan
    mean_reasoning_steps = float(np.mean([row["reasoning_steps"] for row in rows])) if rows else math.nan
    early_exit_rate = float(np.mean([row["reasoning_steps"] < model.cfg.reasoning_steps for row in rows])) if rows else math.nan
    return {
        "num_eval_samples": len(rows),
        "mean_abs_error": mean_abs_error,
        "mean_l2_error": mean_l2_error,
        "mean_reasoning_steps": mean_reasoning_steps,
        "early_exit_rate": early_exit_rate,
        "samples": rows,
    }


def calibrate_exit_threshold(
    model: LightweightAVAVLA,
    samples: List[Dict[str, Any]],
    device: torch.device,
    target_rate: Optional[float],
    logger: logging.Logger,
) -> float:
    if target_rate is None:
        logger.info("Keeping paper-calibrated exit_threshold=%.2f", model.cfg.exit_threshold)
        return model.cfg.exit_threshold

    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for sample in samples:
            observation = torch.from_numpy(sample["observation"]).unsqueeze(0).to(device)
            outputs = model(observation, training=False, adaptive_exit=False, fixed_steps=model.cfg.reasoning_steps)
            exit_scores = outputs["exit_scores"].squeeze(0)
            candidate_scores = exit_scores[:-1] if exit_scores.numel() > 1 else exit_scores
            scores.extend(float(score) for score in candidate_scores.detach().cpu())

    if not scores:
        return model.cfg.exit_threshold

    quantile = float(np.clip(1.0 - target_rate, 0.0, 1.0))
    threshold = float(np.quantile(np.asarray(scores, dtype=np.float32), quantile))
    threshold = float(np.clip(threshold - 1e-4, 0.05, 0.95))
    model.cfg.exit_threshold = threshold
    logger.info("Calibrated exit_threshold=%.6f for target early-exit rate %.2f", threshold, target_rate)
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord",
        default="data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016",
    )
    parser.add_argument("--samples-json", default=None, help="Optional pre-extracted 10-sample JSON dataset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reasoning-policy-type", choices=["softmax", "gaussian"], default="gaussian")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bc-steps", type=int, default=10)
    parser.add_argument("--latent-warmup-steps", type=int, default=5)
    parser.add_argument("--ppo-steps", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=4)
    parser.add_argument("--exit-calibration-steps", type=int, default=5)
    parser.add_argument(
        "--exit-calibration-target-rate",
        type=float,
        default=None,
        help="Optional smoke-test quantile calibration; omitted by default to keep the paper threshold 0.55",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Backward-compatible alias for --ppo-steps")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"runs/avavla_libero_10sample/{timestamp}")
    cfg = AVAVLALiberoTrainConfig(
        tfrecord=args.tfrecord,
        output_dir=str(output_dir),
        samples_json=args.samples_json,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reasoning_policy_type=args.reasoning_policy_type,
        bc_steps=args.bc_steps,
        latent_warmup_steps=args.latent_warmup_steps,
        ppo_steps=args.iterations if args.iterations is not None else args.ppo_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_minibatch_size=args.ppo_minibatch_size,
        exit_calibration_steps=args.exit_calibration_steps,
        exit_calibration_target_rate=args.exit_calibration_target_rate,
        device=args.device,
    )
    logger = setup_logging(output_dir)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")

    if cfg.samples_json is not None:
        samples = load_samples_json(Path(cfg.samples_json), limit=cfg.train_samples)
        sample_source = cfg.samples_json
    else:
        samples = load_libero_samples(Path(cfg.tfrecord), limit=cfg.train_samples)
        sample_source = cfg.tfrecord
    save_samples_json(samples, output_dir / "samples_10.json")
    input_dim = int(samples[0]["observation"].shape[0])
    action_dim = int(samples[0]["action"].shape[0])
    logger.info("Output directory: %s", output_dir)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))
    logger.info("Loaded exactly %d step samples from %s", len(samples), sample_source)
    logger.info("Input dim: %d | Action dim: %d | Device: %s", input_dim, action_dim, device)

    (output_dir / "config.json").write_text(json.dumps(asdict(cfg) | {"input_dim": input_dim, "action_dim": action_dim}, indent=2))
    dataset = LiberoStepDataset(samples)
    generator = torch.Generator().manual_seed(cfg.seed)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, generator=generator)

    model = LightweightAVAVLA(input_dim=input_dim, cfg=cfg, action_dim=action_dim).to(device)
    metrics_path = output_dir / "metrics.jsonl"
    started = time.time()
    global_step = 0
    for stage, steps in (
        ("bc", cfg.bc_steps),
        ("latent_warmup", cfg.latent_warmup_steps),
        ("ppo", cfg.ppo_steps),
        ("exit_calibration", cfg.exit_calibration_steps),
    ):
        global_step = train_stage(model, dataloader, stage, steps, cfg, device, logger, metrics_path, global_step)

    calibrated_threshold = calibrate_exit_threshold(
        model,
        samples,
        device,
        cfg.exit_calibration_target_rate,
        logger,
    )
    cfg.exit_threshold = calibrated_threshold
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg) | {"input_dim": input_dim, "action_dim": action_dim}, indent=2))

    eval_results = evaluate(model, samples, device, cfg.eval_samples)
    eval_path = output_dir / "eval_results.json"
    eval_path.write_text(json.dumps(eval_results, indent=2))

    checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pth"
    torch.save(
        {
            "step": global_step,
            "training_stages": {
                "bc_steps": cfg.bc_steps,
                "latent_warmup_steps": cfg.latent_warmup_steps,
                "ppo_steps": cfg.ppo_steps,
                "exit_calibration_steps": cfg.exit_calibration_steps,
            },
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "input_dim": input_dim,
            "action_dim": action_dim,
            "eval_results": eval_results,
        },
        checkpoint_path,
    )
    logger.info("Finished in %.2fs", time.time() - started)
    logger.info("Saved checkpoint: %s", checkpoint_path)
    logger.info("Saved eval results: %s", eval_path)
    logger.info("Saved metrics: %s", metrics_path)
    logger.info(
        "Eval mean_abs_error=%.6f mean_l2_error=%.6f mean_steps=%.2f early_exit=%.2f",
        eval_results["mean_abs_error"],
        eval_results["mean_l2_error"],
        eval_results["mean_reasoning_steps"],
        eval_results["early_exit_rate"],
    )


if __name__ == "__main__":
    main()
