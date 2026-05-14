"""
avavla.py

PyTorch modules for AVA-VLA: latent reasoning as a POMDP, RL-based reasoning denoising, and adaptive
early-exit for Vision-Language-Action models.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.vlms.prismatic import IGNORE_INDEX, PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool sequence features with an optional attention mask."""
    if mask is None:
        return values.mean(dim=1)

    mask = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (values * mask).sum(dim=1) / denom


def _match_last_dim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or truncate the final dimension so optional history features can be projected consistently."""
    current_dim = values.shape[-1]
    if current_dim == target_dim:
        return values
    if current_dim > target_dim:
        return values[..., :target_dim]

    pad = values.new_zeros(*values.shape[:-1], target_dim - current_dim)
    return torch.cat([values, pad], dim=-1)


class ReasoningPolicy(nn.Module):
    """
    Reasoning policy π_phi(u_t | z_t, o_t).

    The default policy is a Softmax over latent update modes. A conditional Gaussian mode is available for
    continuous update-action experiments.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 1024,
        update_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        policy_type: str = "softmax",
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.update_dim = update_dim
        self.policy_type = policy_type
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.input_proj = nn.Linear(latent_dim + obs_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        output_head = [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        if policy_type == "gaussian":
            self.output_mean = nn.Sequential(*output_head, nn.Linear(hidden_dim, update_dim))
            self.output_log_std = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, update_dim),
            )
        elif policy_type == "softmax":
            self.output_logits = nn.Sequential(*output_head, nn.Linear(hidden_dim, update_dim))
        else:
            raise ValueError(f"Unsupported reasoning policy type: {policy_type}")

    def forward(self, z_t: torch.Tensor, o_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the conditional update-action distribution parameters.

        Args:
            z_t: Latent reasoning state [B, latent_dim]
            o_t: Multimodal observation encoding [B, obs_dim]

        Returns:
            policy_output: Distribution parameters for either Gaussian or Softmax mode.
        """
        combined = torch.cat([z_t, o_t], dim=-1)
        token = self.input_proj(combined).unsqueeze(1)
        hidden = self.transformer(token).squeeze(1)
        if self.policy_type == "softmax":
            logits = self.output_logits(hidden)
            return {
                "logits": logits,
                "probs": F.softmax(logits, dim=-1),
            }

        mean = self.output_mean(hidden)
        log_std = self.output_log_std(hidden).clamp(self.min_log_std, self.max_log_std)
        return {
            "mean": mean,
            "log_std": log_std,
            "std": log_std.exp(),
        }

    def _distribution(self, policy_output: Dict[str, torch.Tensor]):
        if "mean" in policy_output:
            return torch.distributions.Independent(
                torch.distributions.Normal(policy_output["mean"], policy_output["std"]),
                1,
            )
        return torch.distributions.Categorical(probs=policy_output["probs"])

    def sample_update_action(
        self,
        policy_output: Dict[str, torch.Tensor],
        training: bool,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or greedily select a latent update action.

        Returns:
            update_action: One-hot/straight-through update action [B, update_dim]
            log_prob: Log probability of the selected update action [B]
            entropy: Policy entropy [B]
        """
        if "mean" in policy_output:
            dist = self._distribution(policy_output)
            update_action = dist.rsample() if training else policy_output["mean"]
            log_prob = dist.log_prob(update_action)
            entropy = dist.entropy()
            return update_action, log_prob, entropy

        logits, probs = policy_output["logits"], policy_output["probs"]
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if training:
            update_action = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        else:
            indices = probs.argmax(dim=-1)
            update_action = F.one_hot(indices, num_classes=self.update_dim).to(dtype=probs.dtype, device=probs.device)

        log_prob = (update_action.detach() * log_probs).sum(dim=-1)
        return update_action, log_prob, entropy

    def evaluate_update_action(
        self,
        policy_output: Dict[str, torch.Tensor],
        update_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return log probability and entropy for PPO updates on stored update actions."""
        if "mean" in policy_output:
            dist = self._distribution(policy_output)
            return dist.log_prob(update_action), dist.entropy()

        log_probs = F.log_softmax(policy_output["logits"], dim=-1)
        probs = policy_output["probs"]
        entropy = -(probs * log_probs).sum(dim=-1)
        return (update_action * log_probs).sum(dim=-1), entropy


class LatentTransition(nn.Module):
    """Parameterized transition f_theta implemented as GRU + cross-attention with gated increments."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 1024,
        update_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.update_dim = update_dim

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gru = nn.GRUCell(latent_dim + obs_dim, hidden_dim)
        self.update_hidden = nn.Linear(update_dim, hidden_dim)
        self.delta_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(update_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.state_norm = nn.LayerNorm(latent_dim)

    def forward(self, z_t: torch.Tensor, o_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """
        Update latent reasoning state.

        Args:
            z_t: Current latent state [B, latent_dim]
            o_t: Observation encoding [B, obs_dim]
            u_t: Reasoning update action [B, update_dim]

        Returns:
            z_{t+1}: Updated latent state [B, latent_dim]
        """
        latent_token = self.latent_proj(z_t).unsqueeze(1)
        obs_token = self.obs_proj(o_t).unsqueeze(1)
        attended_obs, _ = self.cross_attention(latent_token, obs_token, obs_token, need_weights=False)
        attended_obs = attended_obs.squeeze(1)
        gru_input = torch.cat([z_t, o_t], dim=-1)
        gru_hidden = self.gru(gru_input, self.update_hidden(u_t))

        delta_input = torch.cat([gru_hidden, attended_obs], dim=-1)
        delta_z = self.delta_net(delta_input)
        gate = self.gate_net(u_t)

        return self.state_norm(z_t + gate * delta_z)


class ExitGate(nn.Module):
    """Exit determination function g_omega(z_t), returning state sufficiency in [0, 1]."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z_t))


class ValueFunction(nn.Module):
    """Value function V^π(z_t) used as the Actor-Critic baseline."""

    def __init__(self, latent_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.net(z_t)


class AVAVLA(OpenVLA):
    """
    Adaptive Variable Alignment VLA (AVA-VLA).

    The model augments OpenVLA with:
    - latent reasoning states z_t,
    - a Softmax reasoning policy π_phi over internal update actions,
    - gated incremental transition dynamics f_theta,
    - Actor-Critic RL denoising with entropy and smoothness penalties,
    - adaptive early-exit over the latent reasoning stream,
    - action generation explicitly conditioned on the finalized latent state.
    """

    AVA_PARAMETER_PREFIXES = (
        "visual_obs_proj",
        "text_obs_proj",
        "history_obs_proj",
        "obs_fusion",
        "initial_latent_proj",
        "reasoning_policy",
        "latent_transition",
        "exit_gate",
        "value_function",
        "latent_to_llm",
    )
    AVA_PARAMETER_NAMES = {"latent_action_scale"}

    def __init__(
        self,
        *args,
        latent_dim: int = 512,
        obs_dim: int = 768,
        reasoning_hidden_dim: int = 1024,
        transition_hidden_dim: int = 1024,
        exit_gate_hidden_dim: int = 256,
        value_hidden_dim: int = 512,
        update_dim: int = 64,
        reasoning_policy_type: str = "softmax",
        max_reasoning_steps: int = 5,
        exit_threshold: float = 0.8,
        enable_latent_reasoning: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.update_dim = update_dim
        self.reasoning_policy_type = reasoning_policy_type
        self.max_reasoning_steps = max_reasoning_steps
        self.exit_threshold = exit_threshold
        self.enable_latent_reasoning = enable_latent_reasoning
        self._cached_history_state: Optional[torch.Tensor] = None

        self.vision_dim = getattr(self.vision_backbone, "embed_dim", obs_dim)
        self.llm_dim = getattr(self.llm_backbone, "embed_dim", obs_dim)

        if self.enable_latent_reasoning:
            # ψ(o_t): visual, language, and optional history encodings fused into a unified observation vector.
            self.visual_obs_proj = nn.Linear(self.vision_dim, obs_dim)
            self.text_obs_proj = nn.Linear(self.llm_dim, obs_dim)
            self.history_obs_proj = nn.Linear(latent_dim, obs_dim)
            self.obs_fusion = nn.Sequential(
                nn.LayerNorm(obs_dim * 3),
                nn.Linear(obs_dim * 3, obs_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(obs_dim, obs_dim),
                nn.LayerNorm(obs_dim),
            )

            self.initial_latent_proj = nn.Sequential(
                nn.LayerNorm(obs_dim),
                nn.Linear(obs_dim, latent_dim),
                nn.Tanh(),
            )
            self.reasoning_policy = ReasoningPolicy(
                latent_dim=latent_dim,
                obs_dim=obs_dim,
                hidden_dim=reasoning_hidden_dim,
                update_dim=update_dim,
                policy_type=reasoning_policy_type,
            )
            self.latent_transition = LatentTransition(
                latent_dim=latent_dim,
                obs_dim=obs_dim,
                hidden_dim=transition_hidden_dim,
                update_dim=update_dim,
            )
            self.exit_gate = ExitGate(latent_dim=latent_dim, hidden_dim=exit_gate_hidden_dim)
            self.value_function = ValueFunction(latent_dim=latent_dim, hidden_dim=value_hidden_dim)

            # h_psi([z_t, ψ(o_t)]): inject finalized latent reasoning into the action-generation hidden stream.
            self.latent_to_llm = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, self.llm_dim),
            )
            self.latent_action_scale = nn.Parameter(torch.tensor(0.1))

            if self.all_module_keys is not None:
                self.all_module_keys = self.all_module_keys + [
                    "visual_obs_proj",
                    "text_obs_proj",
                    "history_obs_proj",
                    "obs_fusion",
                    "initial_latent_proj",
                    "reasoning_policy",
                    "latent_transition",
                    "exit_gate",
                    "value_function",
                    "latent_to_llm",
                ]

    @classmethod
    def is_avavla_parameter_name(cls, name: str) -> bool:
        """Return True if a state-dict key belongs to AVA-VLA-specific parameters."""
        return name in cls.AVA_PARAMETER_NAMES or any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in cls.AVA_PARAMETER_PREFIXES
        )

    def get_avavla_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only AVA-VLA-specific parameters so checkpoints do not duplicate the full base VLA."""
        return {name: tensor for name, tensor in self.state_dict().items() if self.is_avavla_parameter_name(name)}

    def load_avavla_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load AVA-VLA-specific parameters from a compact component checkpoint."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if strict:
            expected = set(self.get_avavla_state_dict().keys())
            missing_ava = sorted(name for name in missing if self.is_avavla_parameter_name(name))
            unexpected_ava = sorted(name for name in unexpected if name in expected or self.is_avavla_parameter_name(name))
            if missing_ava or unexpected_ava:
                raise RuntimeError(
                    "Invalid AVA-VLA component checkpoint: "
                    f"missing AVA keys={missing_ava}, unexpected AVA keys={unexpected_ava}"
                )

    def reset_latent_history(self) -> None:
        """Clear cached h_{t-1} used by stateful deployment loops."""
        self._cached_history_state = None

    def _resolve_history_state(
        self,
        history_states: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Use explicit history when provided; otherwise reuse the cached previous latent state."""
        if history_states is not None:
            return history_states
        if self._cached_history_state is None:
            return None
        cached = self._cached_history_state.to(device=device)
        if cached.shape[0] != batch_size:
            return None
        return cached

    def encode_observation(
        self,
        pixel_values: torch.Tensor | Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode o_t = {v_t, l_t, h_{t-1}} into ψ(o_t).

        Args:
            pixel_values: Image tensor or fused-backbone image dict
            input_ids: Text input tokens [B, seq_len]
            attention_mask: Optional language attention mask [B, seq_len]
            history_states: Optional previous latent/history states [B, D] or [B, T, D]

        Returns:
            obs_encoding: Encoded observation [B, obs_dim]
        """
        if not self.enable_latent_reasoning:
            raise RuntimeError("encode_observation requires enable_latent_reasoning=True")

        obs_dtype = self.visual_obs_proj.weight.dtype

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            visual_features = self.vision_backbone(pixel_values)
        visual_encoding = visual_features.mean(dim=1).to(dtype=obs_dtype)
        visual_obs = self.visual_obs_proj(visual_encoding)

        text_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        text_encoding = _masked_mean(text_embeddings, attention_mask).to(dtype=obs_dtype)
        text_obs = self.text_obs_proj(text_encoding)

        if history_states is None:
            history_features = visual_obs.new_zeros(visual_obs.shape[0], self.latent_dim)
        else:
            history_features = history_states
            if history_features.dim() == 3:
                history_features = history_features.mean(dim=1)
            history_features = _match_last_dim(history_features.to(device=visual_obs.device, dtype=obs_dtype), self.latent_dim)
        history_obs = self.history_obs_proj(history_features)

        return self.obs_fusion(torch.cat([visual_obs, text_obs, history_obs], dim=-1))

    def latent_reasoning_forward(
        self,
        z_t: torch.Tensor,
        obs_encoding: torch.Tensor,
        num_steps: Optional[int] = None,
        training: bool = False,
        return_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Run latent reasoning with adaptive early exit.

        During training, all samples run a fixed number of steps so Actor-Critic losses have a dense trajectory.
        During inference, each sample exits independently once g_omega(z_t) > τ.
        """
        if not self.enable_latent_reasoning:
            batch_size = z_t.shape[0]
            empty_scores = z_t.new_zeros(batch_size, 0)
            return z_t, empty_scores, {
                "num_steps_performed": 0,
                "num_steps_per_sample": torch.zeros(batch_size, dtype=torch.long, device=z_t.device),
            }

        max_steps = self.max_reasoning_steps if num_steps is None else num_steps
        if max_steps <= 0:
            batch_size = z_t.shape[0]
            empty_scores = z_t.new_zeros(batch_size, 0)
            return z_t, empty_scores, {
                "num_steps_performed": 0,
                "num_steps_per_sample": torch.zeros(batch_size, dtype=torch.long, device=z_t.device),
            }

        batch_size = z_t.shape[0]
        current_z = z_t
        final_z = current_z
        active = torch.ones(batch_size, dtype=torch.bool, device=z_t.device)
        steps_per_sample = torch.zeros(batch_size, dtype=torch.long, device=z_t.device)

        exit_scores = []
        latent_states = []
        next_latent_states = []
        update_actions = []
        action_log_probs = []
        action_entropies = []
        valid_masks = []

        for _ in range(max_steps):
            step_active = active if not training else torch.ones_like(active)
            valid_masks.append(step_active)

            z_before = current_z
            policy_output = self.reasoning_policy(z_before, obs_encoding)
            u_t, log_prob, entropy = self.reasoning_policy.sample_update_action(policy_output, training=training)
            z_next = self.latent_transition(z_before, obs_encoding, u_t)
            e_t = self.exit_gate(z_next).squeeze(-1)

            if return_trajectory:
                latent_states.append(z_before)
                next_latent_states.append(z_next)
                update_actions.append(u_t)
                action_log_probs.append(log_prob)
                action_entropies.append(entropy)
            exit_scores.append(e_t)

            if training:
                current_z = z_next
                steps_per_sample += 1
                continue

            current_z = torch.where(step_active.unsqueeze(-1), z_next, current_z)
            steps_per_sample += step_active.long()

            newly_finished = step_active & (e_t > self.exit_threshold)
            final_z = torch.where(newly_finished.unsqueeze(-1), current_z, final_z)
            active = step_active & ~newly_finished
            if not active.any():
                break

        if training:
            final_z = current_z
        else:
            final_z = torch.where(active.unsqueeze(-1), current_z, final_z)

        exit_scores_tensor = torch.stack(exit_scores, dim=1) if exit_scores else z_t.new_zeros(batch_size, 0)
        reasoning_info: Dict[str, torch.Tensor | int] = {
            "num_steps_performed": int(exit_scores_tensor.shape[1]),
            "num_steps_per_sample": steps_per_sample,
            "final_exit_scores": exit_scores_tensor[:, -1] if exit_scores_tensor.numel() else z_t.new_zeros(batch_size),
        }

        if return_trajectory and latent_states:
            reasoning_info.update(
                {
                    "latent_states": torch.stack(latent_states, dim=1),
                    "next_latent_states": torch.stack(next_latent_states, dim=1),
                    "update_actions": torch.stack(update_actions, dim=1),
                    "action_log_probs": torch.stack(action_log_probs, dim=1),
                    "old_action_log_probs": torch.stack(action_log_probs, dim=1).detach(),
                    "action_entropies": torch.stack(action_entropies, dim=1),
                    "exit_scores": exit_scores_tensor,
                    "valid_mask": torch.stack(valid_masks, dim=1).to(dtype=obs_encoding.dtype),
                    "obs_encodings": obs_encoding,
                }
            )

        return final_z, exit_scores_tensor, reasoning_info

    def compute_rl_loss(
        self,
        reasoning_trajectories: Dict,
        rewards: torch.Tensor,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        smoothness_coef: float = 0.1,
        value_coef: float = 0.5,
        exit_loss_coef: float = 0.1,
        exit_targets: Optional[torch.Tensor] = None,
        ppo_clip_ratio: float = 0.2,
        gae_lambda: float = 0.95,
        recompute_policy: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute RL-based latent reasoning denoising loss.

        Implements the AVA-VLA composite reward:
            r_t = r_task(a_t) - λ1 H(π_phi(.|z_t,o_t)) - λ2 ||z_{t+1} - z_t||^2
        and optimizes π_phi with PPO-style clipped Actor-Critic and GAE.
        """
        if not self.enable_latent_reasoning or "latent_states" not in reasoning_trajectories:
            return rewards.new_tensor(0.0), {}

        latent_states = reasoning_trajectories["latent_states"]
        next_latent_states = reasoning_trajectories["next_latent_states"]
        stored_log_probs = reasoning_trajectories["action_log_probs"]
        entropies = reasoning_trajectories["action_entropies"]
        old_log_probs = reasoning_trajectories.get("old_action_log_probs", stored_log_probs.detach())
        exit_scores = reasoning_trajectories.get("exit_scores")

        rewards = rewards.to(device=latent_states.device, dtype=latent_states.dtype).view(latent_states.shape[0])
        batch_size, num_steps, _ = latent_states.shape
        log_probs = stored_log_probs.to(device=latent_states.device, dtype=latent_states.dtype)
        valid_mask = reasoning_trajectories.get("valid_mask", torch.ones_like(log_probs)).to(
            device=latent_states.device,
            dtype=latent_states.dtype,
        )
        old_log_probs = old_log_probs.to(device=latent_states.device, dtype=latent_states.dtype).detach()
        entropies = entropies.to(device=latent_states.device, dtype=latent_states.dtype)
        denom = valid_mask.sum().clamp_min(1.0)

        update_actions = reasoning_trajectories.get("update_actions")
        obs_encodings = reasoning_trajectories.get("obs_encodings")
        policy_recomputed = False
        if recompute_policy and update_actions is not None and obs_encodings is not None:
            update_actions = update_actions.to(device=latent_states.device, dtype=latent_states.dtype)
            if obs_encodings.dim() == 2:
                obs_steps = obs_encodings.unsqueeze(1).expand(-1, num_steps, -1)
            else:
                obs_steps = obs_encodings
            obs_steps = obs_steps.to(device=latent_states.device, dtype=latent_states.dtype)

            flat_policy_output = self.reasoning_policy(
                latent_states.reshape(batch_size * num_steps, -1),
                obs_steps.reshape(batch_size * num_steps, -1),
            )
            flat_log_probs, flat_entropies = self.reasoning_policy.evaluate_update_action(
                flat_policy_output,
                update_actions.reshape(batch_size * num_steps, -1),
            )
            log_probs = flat_log_probs.reshape(batch_size, num_steps).to(dtype=latent_states.dtype)
            entropies = flat_entropies.reshape(batch_size, num_steps).to(dtype=latent_states.dtype)
            policy_recomputed = True

        values = self.value_function(latent_states).squeeze(-1)
        smoothness = (next_latent_states - latent_states).pow(2).mean(dim=-1)

        task_rewards = rewards.unsqueeze(1).expand(batch_size, num_steps)
        immediate_rewards = task_rewards - entropy_coef * entropies.detach() - smoothness_coef * smoothness.detach()

        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        next_valid_mask = torch.cat([valid_mask[:, 1:], torch.zeros_like(valid_mask[:, :1])], dim=1)
        deltas = immediate_rewards + gamma * next_values.detach() * next_valid_mask - values.detach()
        advantages = torch.zeros_like(values)
        running_advantage = torch.zeros(batch_size, device=latent_states.device, dtype=latent_states.dtype)
        for step in reversed(range(num_steps)):
            running_advantage = deltas[:, step] + gamma * gae_lambda * running_advantage * next_valid_mask[:, step]
            running_advantage = torch.where(
                valid_mask[:, step].bool(),
                running_advantage,
                torch.zeros_like(running_advantage),
            )
            advantages[:, step] = running_advantage
        returns = advantages + values.detach()
        advantage_mean = (advantages * valid_mask).sum() / denom
        advantage_var = (((advantages - advantage_mean) * valid_mask).pow(2).sum() / denom).clamp_min(1e-8)
        advantages = (advantages - advantage_mean) / advantage_var.sqrt()

        ratio = torch.exp((log_probs - old_log_probs).clamp(-20.0, 20.0))
        unclipped_policy = ratio * advantages.detach()
        clipped_policy = torch.clamp(ratio, 1.0 - ppo_clip_ratio, 1.0 + ppo_clip_ratio) * advantages.detach()
        policy_loss = -((torch.minimum(unclipped_policy, clipped_policy) * valid_mask).sum() / denom)
        value_loss = (((values - returns.detach()).pow(2)) * valid_mask).sum() / denom
        entropy_penalty = (entropies * valid_mask).sum() / denom
        smoothness_loss = (smoothness * valid_mask).sum() / denom
        if recompute_policy:
            exit_scores = self.exit_gate(next_latent_states.reshape(batch_size * num_steps, -1)).reshape(
                batch_size,
                num_steps,
            )
        if exit_scores is not None and exit_scores.numel() > 0:
            exit_scores = exit_scores.to(device=latent_states.device, dtype=latent_states.dtype)
            if exit_targets is None:
                exit_target_tensor = rewards.detach().clamp(0.0, 1.0)
            else:
                exit_target_tensor = exit_targets.to(device=latent_states.device, dtype=latent_states.dtype).view(batch_size)
            exit_target_tensor = exit_target_tensor.unsqueeze(1).expand_as(exit_scores)
            exit_loss = F.binary_cross_entropy(
                exit_scores.clamp(1e-6, 1.0 - 1e-6),
                exit_target_tensor,
                reduction="none",
            )
            exit_loss = (exit_loss * valid_mask).sum() / denom
        else:
            exit_loss = latent_states.new_tensor(0.0)

        # Entropy and smoothness are already part of the composite reward above; adding them again here would
        # silently change the meaning of lambda_1/lambda_2.
        rl_loss = policy_loss + value_coef * value_loss + exit_loss_coef * exit_loss

        return rl_loss, {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "exit_loss": float(exit_loss.detach().cpu()),
            "entropy_penalty": float(entropy_penalty.detach().cpu()),
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "ppo_ratio_mean": float(((ratio * valid_mask).sum() / denom).detach().cpu()),
            "ppo_clip_fraction": float(
                ((((ratio - 1.0).abs() > ppo_clip_ratio).to(dtype=valid_mask.dtype) * valid_mask).sum() / denom)
                .detach()
                .cpu()
            ),
            "ppo_policy_recomputed": float(policy_recomputed),
            "gae_advantage_mean": float(((advantages * valid_mask).sum() / denom).detach().cpu()),
            "mean_composite_reward": float(((immediate_rewards * valid_mask).sum() / denom).detach().cpu()),
            "total_rl_loss": float(rl_loss.detach().cpu()),
        }

    def _fuse_latent_into_embeddings(
        self, input_embeddings: torch.Tensor, latent_state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Condition the action policy on z_t by adding a learned latent bias to language/action embeddings."""
        if latent_state is None or not self.enable_latent_reasoning:
            return input_embeddings

        if latent_state.dim() == 3:
            latent_state = latent_state[:, -1]
        latent_state = latent_state.to(device=input_embeddings.device, dtype=self.latent_to_llm[1].weight.dtype)
        latent_bias = self.latent_to_llm(latent_state).to(dtype=input_embeddings.dtype).unsqueeze(1)
        return input_embeddings + self.latent_action_scale.to(dtype=input_embeddings.dtype) * latent_bias

    def _zero_action_token_embeddings(
        self,
        input_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Remove ground-truth action token identity from the hidden stream used by continuous action heads."""
        if labels is None:
            raise ValueError("zero_action_token_embeddings=True requires labels to identify action-token positions.")

        action_token_begin_idx = self.action_tokenizer.action_token_begin_idx
        action_mask = labels.to(device=input_embeddings.device) > action_token_begin_idx
        return input_embeddings.masked_fill(action_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        latent_state: Optional[torch.Tensor] = None,
        zero_action_token_embeddings: bool = False,
        **kwargs,
    ):
        """Forward pass, optionally injecting finalized latent reasoning state into action generation."""
        # Drop HF-OpenVLA-only kwargs if this local PrismaticVLM wrapper is used by AVA-VLA scripts.
        kwargs.pop("proprio", None)
        kwargs.pop("proprio_projector", None)
        kwargs.pop("noisy_actions", None)
        kwargs.pop("noisy_action_projector", None)
        kwargs.pop("diffusion_timestep_embeddings", None)
        kwargs.pop("use_film", None)

        if (latent_state is None or not self.enable_latent_reasoning) and not zero_action_token_embeddings:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                multimodal_indices=multimodal_indices,
            )

        if input_ids.shape[1] == 1 and past_key_values is not None:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if input_ids.shape[1] == 1 or pixel_values is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                multimodal_indices=multimodal_indices,
            )

        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        elif len(multimodal_indices) == 0:
            input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
            if zero_action_token_embeddings:
                input_embeddings = self._zero_action_token_embeddings(input_embeddings, labels)
            input_embeddings = self._fuse_latent_into_embeddings(input_embeddings, latent_state)
            return self.llm_backbone(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=input_embeddings,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        if zero_action_token_embeddings:
            input_embeddings = self._zero_action_token_embeddings(input_embeddings, labels)
        input_embeddings = self._fuse_latent_into_embeddings(input_embeddings, latent_state)

        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )

        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            if labels is None or attention_mask is None:
                raise RuntimeError("Mixed multimodal/unimodal AVA-VLA batches require labels and attention_mask.")

            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if "latent_state" in kwargs:
            model_inputs["latent_state"] = kwargs["latent_state"]
        return model_inputs

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        num_reasoning_steps: Optional[int] = None,
        return_reasoning_info: bool = False,
        history_states: Optional[torch.Tensor] = None,
        update_history: bool = True,
        **kwargs,
    ) -> np.ndarray | Tuple[np.ndarray, Dict]:
        """Predict an action; when latent reasoning is enabled, condition generation on the finalized z_t."""
        if self.enable_latent_reasoning:
            actions, info = self.predict_action_with_latent_reasoning(
                image=image,
                instruction=instruction,
                unnorm_key=unnorm_key,
                num_reasoning_steps=num_reasoning_steps,
                history_states=history_states,
                update_history=update_history,
                **kwargs,
            )
            return (actions, info) if return_reasoning_info else actions

        actions = super().predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        if return_reasoning_info:
            return actions, {"num_steps_performed": 0, "exit_scores_history": None}
        return actions

    @torch.inference_mode()
    def predict_action_with_latent_reasoning(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        num_reasoning_steps: Optional[int] = None,
        history_states: Optional[torch.Tensor] = None,
        update_history: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """Predict action and return latent-reasoning diagnostics."""
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        history_states = self._resolve_history_state(history_states, input_ids.shape[0], input_ids.device)
        obs_encoding = self.encode_observation(
            pixel_values,
            input_ids,
            attention_mask=attention_mask,
            history_states=history_states,
        )
        z_0 = self.initial_latent_proj(obs_encoding)
        final_z, exit_scores, reasoning_info = self.latent_reasoning_forward(
            z_0,
            obs_encoding,
            num_steps=num_reasoning_steps,
            training=False,
            return_trajectory=False,
        )

        autocast_dtype = self.llm_backbone.half_precision_dtype
        autocast_enabled = self.enable_mixed_precision_training and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                latent_state=final_z,
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs,
            )

        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        reasoning_info["exit_threshold"] = self.exit_threshold
        reasoning_info["exit_scores_history"] = exit_scores.detach().cpu().numpy()
        reasoning_info["num_steps_per_sample"] = reasoning_info["num_steps_per_sample"].detach().cpu().numpy()
        if update_history:
            self._cached_history_state = final_z.detach()
        return actions, reasoning_info
