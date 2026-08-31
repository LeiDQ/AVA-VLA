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
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
from prismatic.vla.preprocessing import center_crop_for_augmentation

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


def _floating_parameter_dtype(module: object, fallback: torch.dtype) -> torch.dtype:
    """Return a module's floating compute dtype without assuming it is an nn.Module.

    Online trajectories are intentionally stored in BF16, while the small PPO
    heads remain FP32 for stability.  Explicitly matching inputs to the owning
    module avoids relying on an ambient autocast context during PPO replay.
    """
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            if torch.is_floating_point(parameter):
                return parameter.dtype
    return fallback


class ReasoningPolicy(nn.Module):
    """
    Reasoning policy π_phi(u_t | z_t, o_t).

    The paper-default policy is a conditional Gaussian over continuous latent update actions. A Softmax mode
    remains available for ablations.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 512,
        update_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        policy_type: str = "gaussian",
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

        # Keep state and observation as distinct tokens.  A one-token Transformer
        # reduces self-attention to an MLP and cannot implement the architecture
        # described in Appendix B.2.
        self.latent_input_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_input_proj = nn.Linear(obs_dim, hidden_dim)
        self.token_type_embeddings = nn.Parameter(torch.zeros(2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        output_head = [
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if policy_type == "gaussian":
            self.output_mean = nn.Sequential(*output_head, nn.Linear(hidden_dim, update_dim))
            self.output_log_std = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
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
        tokens = torch.stack(
            [self.latent_input_proj(z_t), self.obs_input_proj(o_t)],
            dim=1,
        )
        tokens = tokens + self.token_type_embeddings.unsqueeze(0).to(dtype=tokens.dtype)
        hidden = self.transformer(tokens)[:, 0]
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

    def uncertainty_penalty(self, policy_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a non-negative uncertainty measure for the composite reward.

        Differential entropy of a continuous Gaussian becomes negative as its
        variance shrinks.  Subtracting that raw value from reward would then
        *reward* variance collapse.  The paper uses entropy to penalize
        excessive uncertainty, so for the continuous policy we average the
        positive per-dimension differential entropy.  Categorical entropy is
        already non-negative and needs no adjustment.
        """
        if "mean" in policy_output:
            per_dimension = torch.distributions.Normal(
                policy_output["mean"], policy_output["std"]
            ).entropy()
            return per_dimension.clamp_min(0.0).mean(dim=-1)
        return torch.distributions.Categorical(probs=policy_output["probs"]).entropy()

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
            entropy = self.uncertainty_penalty(policy_output)
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
            return dist.log_prob(update_action), self.uncertainty_penalty(policy_output)

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
        num_obs_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.update_dim = update_dim
        self.num_obs_tokens = num_obs_tokens

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim * num_obs_tokens)
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
        obs_tokens = self.obs_proj(o_t).reshape(o_t.shape[0], self.num_obs_tokens, -1)
        attended_obs, _ = self.cross_attention(latent_token, obs_tokens, obs_tokens, need_weights=False)
        attended_obs = attended_obs.squeeze(1)
        gru_input = torch.cat([z_t, o_t], dim=-1)

        # PyTorch 2.2 does not implement the fused CUDA GRUCell kernel for BF16.
        # Keep the recurrent state update in FP32 for numerical stability while
        # allowing the surrounding Transformer/MLPs to remain under BF16 autocast.
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            gru_hidden = self.gru(
                gru_input.float(),
                self.update_hidden(u_t.float()).float(),
            )

        delta_input = torch.cat([gru_hidden, attended_obs], dim=-1)
        delta_z = self.delta_net(delta_input)
        gate = self.gate_net(u_t)

        return self.state_norm(z_t + gate * delta_z)


class _ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP used by the paper's exit gate."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class ExitGate(nn.Module):
    """Exit determination function g_omega(z_t), returning state sufficiency in [0, 1]."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual = _ResidualMLPBlock(hidden_dim, dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output(self.residual(self.input(z_t))))


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
    - a continuous Gaussian reasoning policy π_phi over internal update actions,
    - gated incremental transition dynamics f_theta,
    - Actor-Critic RL denoising with entropy and smoothness penalties,
    - adaptive early-exit over the latent reasoning stream,
    - action generation explicitly conditioned on the finalized latent state.
    """

    AVA_PARAMETER_PREFIXES = (
        "visual_obs_proj",
        "projector",
        "visual_view_fusion",
        "text_obs_proj",
        "proprio_obs_proj",
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
        reasoning_hidden_dim: int = 512,
        reasoning_num_heads: int = 8,
        reasoning_num_layers: int = 4,
        transition_hidden_dim: int = 1024,
        exit_gate_hidden_dim: int = 256,
        value_hidden_dim: int = 512,
        update_dim: int = 64,
        proprio_dim: int = 8,
        dropout: float = 0.1,
        reasoning_policy_type: str = "gaussian",
        max_reasoning_steps: int = 5,
        exit_threshold: float = 0.55,
        enable_latent_reasoning: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.update_dim = update_dim
        self.proprio_dim = proprio_dim
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
            self.visual_view_fusion = nn.Sequential(
                nn.LayerNorm(self.vision_dim * 2),
                nn.Linear(self.vision_dim * 2, self.vision_dim),
                nn.GELU(),
            )
            self.text_obs_proj = nn.Linear(self.llm_dim, obs_dim)
            self.proprio_obs_proj = nn.Sequential(
                nn.LayerNorm(proprio_dim),
                nn.Linear(proprio_dim, obs_dim),
                nn.GELU(),
                nn.Linear(obs_dim, obs_dim),
            )
            self.history_obs_proj = nn.Linear(latent_dim, obs_dim)
            self.obs_fusion = nn.Sequential(
                nn.LayerNorm(obs_dim * 4),
                nn.Linear(obs_dim * 4, obs_dim),
                nn.GELU(),
                nn.Dropout(dropout),
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
                num_heads=reasoning_num_heads,
                num_layers=reasoning_num_layers,
                dropout=dropout,
                policy_type=reasoning_policy_type,
            )
            if reasoning_policy_type == "gaussian":
                # Appendix B.3 initializes the reasoning policy with zero-mean Gaussian updates.
                nn.init.zeros_(self.reasoning_policy.output_mean[-1].weight)
                nn.init.zeros_(self.reasoning_policy.output_mean[-1].bias)
            self.latent_transition = LatentTransition(
                latent_dim=latent_dim,
                obs_dim=obs_dim,
                hidden_dim=transition_hidden_dim,
                update_dim=update_dim,
                dropout=dropout,
            )
            self.exit_gate = ExitGate(latent_dim=latent_dim, hidden_dim=exit_gate_hidden_dim, dropout=dropout)
            self.value_function = ValueFunction(latent_dim=latent_dim, hidden_dim=value_hidden_dim, dropout=dropout)

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
                    "visual_view_fusion",
                    "text_obs_proj",
                    "proprio_obs_proj",
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

    def normalize_proprio(
        self,
        proprio: np.ndarray,
        unnorm_key: Optional[str],
    ) -> np.ndarray:
        """Normalize raw proprioception with the exact RLDS training statistics."""
        if unnorm_key is None:
            if len(self.norm_stats) != 1:
                raise ValueError("unnorm_key is required when multiple dataset statistics are loaded")
            unnorm_key = next(iter(self.norm_stats))
        if unnorm_key not in self.norm_stats:
            raise KeyError(f"Missing normalization statistics for {unnorm_key!r}")
        proprio_stats = self.norm_stats[unnorm_key].get("proprio")
        if proprio_stats is None:
            raise KeyError(f"Missing proprio statistics for {unnorm_key!r}")

        values = np.asarray(proprio, dtype=np.float32)
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            low = np.asarray(proprio_stats["min"], dtype=np.float32)
            high = np.asarray(proprio_stats["max"], dtype=np.float32)
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            low = np.asarray(proprio_stats["q01"], dtype=np.float32)
            high = np.asarray(proprio_stats["q99"], dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported proprio normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
            )
        mask = np.asarray(proprio_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        scaled = 2.0 * (values - low) / (high - low + 1e-8) - 1.0
        normalized = np.where(mask, np.clip(scaled, -1.0, 1.0), values)
        return normalized.astype(np.float32)

    def fuse_observation_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply trainable multimodal fusion to cached, action-free backbone features.

        Online PPO stores these compact frozen-backbone features and rebuilds
        the observation encoding on every PPO replay.  This keeps PPO gradients
        connected to the multimodal projections and initial latent state without
        repeatedly executing the frozen vision and Llama backbones.
        """
        obs_dtype = self.visual_obs_proj.weight.dtype
        primary_encoding = features["primary_visual"].to(dtype=obs_dtype)
        wrist_encoding = features["wrist_visual"].to(
            device=primary_encoding.device,
            dtype=obs_dtype,
        )
        text_encoding = features["text"].to(device=primary_encoding.device, dtype=obs_dtype)
        proprio_features = features["proprio"].to(device=primary_encoding.device, dtype=obs_dtype)
        history_features = features["history"].to(device=primary_encoding.device, dtype=obs_dtype)

        visual_encoding = self.visual_view_fusion(
            torch.cat([primary_encoding, wrist_encoding], dim=-1)
        )
        visual_obs = self.visual_obs_proj(visual_encoding)
        text_obs = self.text_obs_proj(text_encoding)
        proprio_obs = self.proprio_obs_proj(
            _match_last_dim(proprio_features, self.proprio_dim)
        )
        history_obs = self.history_obs_proj(
            _match_last_dim(history_features, self.latent_dim)
        )
        return self.obs_fusion(
            torch.cat([visual_obs, text_obs, proprio_obs, history_obs], dim=-1)
        )

    def encode_observation(
        self,
        pixel_values: torch.Tensor | Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
        pixel_values_wrist: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None,
        proprio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            primary_features = self.vision_backbone(pixel_values)
            wrist_features = (
                self.vision_backbone(pixel_values_wrist)
                if pixel_values_wrist is not None
                else None
            )
        primary_encoding = primary_features.mean(dim=1).to(dtype=obs_dtype)
        wrist_encoding = (
            wrist_features.mean(dim=1).to(dtype=obs_dtype)
            if wrist_features is not None
            else torch.zeros_like(primary_encoding)
        )
        # Observation encoding must never see demonstration action tokens.  The
        # training sequence contains prompt + ground-truth actions for the OFT
        # action head, so use labels to derive a prompt-only mask and replace all
        # non-prompt token identities before calling the frozen Llama encoder.
        text_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=input_ids.device, dtype=torch.bool)
        )
        if labels is not None:
            text_mask = text_mask & labels.to(device=input_ids.device).eq(IGNORE_INDEX)
        pad_token_id = int(self.llm_backbone.pad_token_id)
        text_input_ids = input_ids.masked_fill(~text_mask, pad_token_id)
        with torch.no_grad():
            text_output = self.llm_backbone.llm.model(
                input_ids=text_input_ids,
                attention_mask=text_mask,
                use_cache=False,
                return_dict=True,
            )
            text_hidden = text_output.last_hidden_state
        text_encoding = _masked_mean(text_hidden, text_mask).to(dtype=obs_dtype)

        if proprio is None:
            proprio_features = primary_encoding.new_zeros(primary_encoding.shape[0], self.proprio_dim)
        else:
            proprio_features = _match_last_dim(
                proprio.to(device=primary_encoding.device, dtype=obs_dtype), self.proprio_dim
            )

        if history_states is None:
            history_features = primary_encoding.new_zeros(primary_encoding.shape[0], self.latent_dim)
        else:
            history_features = history_states
            if history_features.dim() == 3:
                history_features = history_features.mean(dim=1)
            history_features = _match_last_dim(
                history_features.to(device=primary_encoding.device, dtype=obs_dtype),
                self.latent_dim,
            )

        features = {
            "primary_visual": primary_encoding,
            "wrist_visual": wrist_encoding,
            "text": text_encoding,
            "proprio": proprio_features,
            "history": history_features,
        }
        obs_encoding = self.fuse_observation_features(features)
        return (obs_encoding, features) if return_features else obs_encoding

    def latent_reasoning_forward(
        self,
        z_t: torch.Tensor,
        obs_encoding: torch.Tensor,
        num_steps: Optional[int] = None,
        training: bool = False,
        return_trajectory: bool = True,
        force_fixed_steps: bool = False,
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
        # The latent policy, transition, critic and exit gate are small FP32
        # modules.  Run the complete recurrent latent stream in their native
        # dtype even when the surrounding VLA encoder is under BF16 autocast.
        # This gives rollout collection and PPO replay the same numerical
        # contract and prevents five recurrent log-prob errors from compounding.
        latent_dtype = _floating_parameter_dtype(self.reasoning_policy, z_t.dtype)
        current_z = z_t.to(dtype=latent_dtype)
        obs_encoding = obs_encoding.to(dtype=latent_dtype)
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

            # True dynamic batching: after a sample exits, do not execute its
            # policy, transition, or exit gate on subsequent iterations.
            active_indices = torch.nonzero(step_active, as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break
            z_before_active = current_z.index_select(0, active_indices)
            obs_active = obs_encoding.index_select(0, active_indices)
            with torch.autocast(device_type=z_t.device.type, enabled=False):
                policy_output = self.reasoning_policy(z_before_active, obs_active)
                u_active, log_prob_active, entropy_active = self.reasoning_policy.sample_update_action(
                    policy_output,
                    training=training,
                )
                z_next_active = self.latent_transition(z_before_active, obs_active, u_active)
                e_active = self.exit_gate(z_next_active).squeeze(-1)
            z_next_active = z_next_active.to(dtype=current_z.dtype)

            z_before = current_z
            z_next = current_z.clone().index_copy(0, active_indices, z_next_active)
            e_t = e_active.new_zeros(batch_size).index_copy(0, active_indices, e_active)
            u_t = u_active.new_zeros(batch_size, self.update_dim).index_copy(0, active_indices, u_active)
            log_prob = log_prob_active.new_zeros(batch_size).index_copy(0, active_indices, log_prob_active)
            entropy = entropy_active.new_zeros(batch_size).index_copy(0, active_indices, entropy_active)

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

            current_z = z_next
            steps_per_sample += step_active.long()

            newly_finished = (
                torch.zeros_like(step_active)
                if force_fixed_steps
                else step_active & (e_t > self.exit_threshold)
            )
            final_z = torch.where(newly_finished.unsqueeze(-1), current_z, final_z)
            active = step_active & ~newly_finished
            if not active.any():
                break

        if training:
            final_z = current_z
        else:
            final_z = torch.where(active.unsqueeze(-1), current_z, final_z)

        exit_scores_tensor = torch.stack(exit_scores, dim=1) if exit_scores else z_t.new_zeros(batch_size, 0)
        if exit_scores_tensor.numel():
            last_step_indices = (steps_per_sample - 1).clamp_min(0).unsqueeze(1)
            final_exit_scores = exit_scores_tensor.gather(1, last_step_indices).squeeze(1)
        else:
            final_exit_scores = z_t.new_zeros(batch_size)
        reasoning_info: Dict[str, torch.Tensor | int] = {
            "num_steps_performed": int(exit_scores_tensor.shape[1]),
            "num_steps_per_sample": steps_per_sample,
            "final_exit_scores": final_exit_scores,
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
        recompute_dynamics: bool = True,
        recompute_observation: bool = True,
        train_exit_gate: bool = False,
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
        compute_dtype = _floating_parameter_dtype(
            getattr(self, "reasoning_policy", None),
            _floating_parameter_dtype(self.value_function, latent_states.dtype),
        )
        # Rollouts are compact BF16 tensors. PPO heads are deliberately FP32;
        # replay them in the head dtype so LayerNorm/Linear kernels agree.
        latent_states = latent_states.to(dtype=compute_dtype)
        next_latent_states = reasoning_trajectories["next_latent_states"].to(dtype=compute_dtype)
        stored_log_probs = reasoning_trajectories["action_log_probs"].to(dtype=compute_dtype)
        entropies = reasoning_trajectories["action_entropies"].to(dtype=compute_dtype)
        old_log_probs = reasoning_trajectories.get("old_action_log_probs", stored_log_probs.detach())
        exit_scores = reasoning_trajectories.get("exit_scores")

        batch_size, num_steps, _ = latent_states.shape
        rewards = rewards.to(device=latent_states.device, dtype=latent_states.dtype)
        if rewards.dim() == 1:
            rewards = rewards.view(batch_size)
        elif rewards.shape != (batch_size, num_steps):
            raise ValueError(
                f"Expected PPO rewards with shape {(batch_size,)} or {(batch_size, num_steps)}, "
                f"got {tuple(rewards.shape)}"
            )
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
        dynamics_recomputed = False
        observation_recomputed = False
        feature_names = ("primary_visual", "wrist_visual", "text", "proprio", "history")
        feature_keys = {name: f"ppo_observation_{name}" for name in feature_names}
        if recompute_observation and all(key in reasoning_trajectories for key in feature_keys.values()):
            observation_features = {
                name: reasoning_trajectories[key].to(device=latent_states.device)
                for name, key in feature_keys.items()
            }
            obs_encodings = self.fuse_observation_features(observation_features)
            initial_latent = self.initial_latent_proj(obs_encodings).to(dtype=latent_states.dtype)
            observation_recomputed = True
        else:
            # Compatibility fallback for legacy/offline unit-test rollouts.
            initial_latent = latent_states[:, 0].detach()
        if recompute_policy and update_actions is not None and obs_encodings is not None:
            update_actions = update_actions.to(device=latent_states.device, dtype=latent_states.dtype)
            if obs_encodings.dim() == 2:
                obs_steps = obs_encodings.unsqueeze(1).expand(-1, num_steps, -1)
            else:
                obs_steps = obs_encodings
            obs_steps = obs_steps.to(device=latent_states.device, dtype=latent_states.dtype)

            if recompute_dynamics:
                current_z = initial_latent
                recomputed_states, recomputed_next_states = [], []
                recomputed_log_probs, recomputed_entropies = [], []
                for step in range(num_steps):
                    recomputed_states.append(current_z)
                    policy_output = self.reasoning_policy(current_z, obs_steps[:, step])
                    step_log_prob, step_entropy = self.reasoning_policy.evaluate_update_action(
                        policy_output, update_actions[:, step]
                    )
                    next_z = self.latent_transition(current_z, obs_steps[:, step], update_actions[:, step])
                    recomputed_next_states.append(next_z)
                    recomputed_log_probs.append(step_log_prob)
                    recomputed_entropies.append(step_entropy)
                    current_z = next_z
                latent_states = torch.stack(recomputed_states, dim=1)
                next_latent_states = torch.stack(recomputed_next_states, dim=1)
                log_probs = torch.stack(recomputed_log_probs, dim=1)
                entropies = torch.stack(recomputed_entropies, dim=1)
                dynamics_recomputed = True
            else:
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

        value_dtype = _floating_parameter_dtype(self.value_function, latent_states.dtype)
        values = self.value_function(latent_states.to(dtype=value_dtype)).squeeze(-1).to(
            dtype=latent_states.dtype
        )
        smoothness = (next_latent_states - latent_states).pow(2).mean(dim=-1)

        if rewards.dim() == 2:
            task_rewards = rewards
        else:
            # A LIBERO outcome belongs to the action chunk produced after the
            # final latent update.  Place it only on the last valid reasoning
            # step; GAE propagates sparse credit backward instead of counting
            # the same success reward once per latent iteration.
            task_rewards = torch.zeros_like(values)
            last_valid = valid_mask.sum(dim=1).long().clamp_min(1) - 1
            task_rewards.scatter_(1, last_valid.unsqueeze(1), rewards.unsqueeze(1))
        stored_step_rewards = reasoning_trajectories.get("ppo_step_rewards")
        if stored_step_rewards is not None:
            immediate_rewards = stored_step_rewards.to(device=values.device, dtype=values.dtype)
        else:
            immediate_rewards = (
                task_rewards
                - entropy_coef * entropies.detach()
                - smoothness_coef * smoothness.detach()
            )

        stored_advantages = reasoning_trajectories.get("ppo_advantages")
        stored_returns = reasoning_trajectories.get("ppo_returns")
        if stored_advantages is not None and stored_returns is not None:
            advantages = stored_advantages.to(device=values.device, dtype=values.dtype)
            returns = stored_returns.to(device=values.device, dtype=values.dtype)
        else:
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
        log_ratio = log_probs - old_log_probs
        approx_kl = ((((ratio - 1.0) - log_ratio) * valid_mask).sum() / denom).clamp_min(0.0)
        unclipped_policy = ratio * advantages.detach()
        clipped_policy = torch.clamp(ratio, 1.0 - ppo_clip_ratio, 1.0 + ppo_clip_ratio) * advantages.detach()
        policy_loss = -((torch.minimum(unclipped_policy, clipped_policy) * valid_mask).sum() / denom)
        value_loss = (((values - returns.detach()).pow(2)) * valid_mask).sum() / denom
        entropy_penalty = (entropies * valid_mask).sum() / denom
        smoothness_loss = (smoothness * valid_mask).sum() / denom
        if train_exit_gate and recompute_policy:
            exit_dtype = _floating_parameter_dtype(self.exit_gate, latent_states.dtype)
            exit_scores = self.exit_gate(
                next_latent_states.reshape(batch_size * num_steps, -1).to(dtype=exit_dtype)
            ).reshape(batch_size, num_steps).to(
                dtype=latent_states.dtype
            )
        if train_exit_gate and exit_scores is not None and exit_scores.numel() > 0:
            exit_scores = exit_scores.to(device=latent_states.device, dtype=latent_states.dtype)
            if exit_targets is None:
                exit_target_tensor = (
                    rewards[:, -1] if rewards.dim() == 2 else rewards
                ).detach().clamp(0.0, 1.0)
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

        # Appendix B.3 defines entropy and smoothness inside the composite
        # reward used by GAE.  Do not add them a second time as direct losses.
        # Stage 2 already provides the explicit differentiable smoothness
        # warmup; Stage 3 optimizes the frozen composite-reward targets once.
        rl_loss = policy_loss + value_coef * value_loss + exit_loss_coef * exit_loss

        return rl_loss, {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "exit_loss": float(exit_loss.detach().cpu()),
            "entropy_penalty": float(entropy_penalty.detach().cpu()),
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "ppo_ratio_mean": float(((ratio * valid_mask).sum() / denom).detach().cpu()),
            "ppo_approx_kl": float(approx_kl.detach().cpu()),
            "ppo_clip_fraction": float(
                ((((ratio - 1.0).abs() > ppo_clip_ratio).to(dtype=valid_mask.dtype) * valid_mask).sum() / denom)
                .detach()
                .cpu()
            ),
            "ppo_policy_recomputed": float(policy_recomputed),
            "ppo_dynamics_recomputed": float(dynamics_recomputed),
            "ppo_observation_recomputed": float(observation_recomputed),
            "regularizers_in_reward_only": 1.0,
            "gae_advantage_mean": float(((advantages * valid_mask).sum() / denom).detach().cpu()),
            "mean_composite_reward": float(((immediate_rewards * valid_mask).sum() / denom).detach().cpu()),
            "total_rl_loss": float(rl_loss.detach().cpu()),
        }

    def compute_latent_warmup_loss(self, reasoning_trajectories: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Stage-2 smoothness objective with gradients into policy and transition dynamics."""
        latent_states = reasoning_trajectories["latent_states"]
        next_latent_states = reasoning_trajectories["next_latent_states"]
        valid_mask = reasoning_trajectories.get(
            "valid_mask", latent_states.new_ones(latent_states.shape[:2])
        ).to(dtype=latent_states.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        per_step = (next_latent_states - latent_states).pow(2).mean(dim=-1)
        loss = (per_step * valid_mask).sum() / denom
        return loss, {
            "latent_warmup_loss": float(loss.detach().cpu()),
            "latent_distance_mean": float((per_step * valid_mask).sum().detach().cpu() / denom.detach().cpu()),
        }

    def compute_exit_calibration_loss(
        self,
        reasoning_trajectories: Dict,
        lookahead: int = 3,
        delta: float = 0.05,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calibrate g_omega after PPO using critic-estimated value of computation.

        A state is a positive exit example when continuing for ``lookahead``
        latent steps improves the learned task value by less than ``delta``.
        All policy, transition, and critic inputs are detached so only the exit
        gate receives gradients during stage 4.
        """
        raw_states = reasoning_trajectories["next_latent_states"].detach()
        exit_dtype = _floating_parameter_dtype(self.exit_gate, raw_states.dtype)
        states = raw_states.to(dtype=exit_dtype)
        batch_size, num_steps, latent_dim = states.shape
        valid_mask = reasoning_trajectories.get(
            "valid_mask", states.new_ones(batch_size, num_steps)
        ).to(dtype=states.dtype)
        with torch.no_grad():
            value_dtype = _floating_parameter_dtype(self.value_function, states.dtype)
            values = self.value_function(
                states.reshape(batch_size * num_steps, latent_dim).to(dtype=value_dtype)
            ).reshape(batch_size, num_steps).to(dtype=states.dtype)
            labels = torch.zeros_like(values)
            for step in range(num_steps):
                future_step = min(step + max(1, int(lookahead)), num_steps - 1)
                improvement = values[:, future_step] - values[:, step]
                labels[:, step] = (improvement < float(delta)).to(dtype=values.dtype)

        scores = self.exit_gate(states.reshape(batch_size * num_steps, latent_dim)).reshape(
            batch_size, num_steps
        )
        per_step_loss = F.binary_cross_entropy(
            scores.clamp(1e-6, 1.0 - 1e-6), labels, reduction="none"
        )
        denom = valid_mask.sum().clamp_min(1.0)
        loss = (per_step_loss * valid_mask).sum() / denom
        accuracy = (
            ((scores >= 0.5) == labels.bool()).to(dtype=valid_mask.dtype) * valid_mask
        ).sum() / denom
        return loss, {
            "exit_calibration_loss": float(loss.detach().cpu()),
            "exit_calibration_accuracy": float(accuracy.detach().cpu()),
            "exit_positive_rate": float(((labels * valid_mask).sum() / denom).detach().cpu()),
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
        initialize_latent_from_observation: bool = False,
        pixel_values_wrist: Optional[torch.FloatTensor | Dict[str, torch.Tensor]] = None,
        proprio: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
        training_objective: Optional[str] = None,
        reasoning_trajectories: Optional[Dict] = None,
        rl_rewards: Optional[torch.Tensor] = None,
        objective_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Forward pass, optionally injecting finalized latent reasoning state into action generation."""
        objective_kwargs = objective_kwargs or {}
        if training_objective == "latent_warmup":
            obs_encoding = self.encode_observation(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                history_states=history_states,
                pixel_values_wrist=pixel_values_wrist,
                proprio=proprio,
                labels=labels,
            )
            initial_z = self.initial_latent_proj(obs_encoding)
            _, _, trajectories = self.latent_reasoning_forward(
                initial_z,
                obs_encoding,
                num_steps=objective_kwargs.get("num_steps", self.max_reasoning_steps),
                training=True,
                return_trajectory=True,
            )
            loss, metrics = self.compute_latent_warmup_loss(trajectories)
            return loss, metrics, trajectories
        if training_objective == "ppo":
            if reasoning_trajectories is None or rl_rewards is None:
                raise ValueError("PPO forward requires reasoning_trajectories and rl_rewards")
            return self.compute_rl_loss(reasoning_trajectories, rl_rewards, **objective_kwargs)
        if training_objective == "exit_calibration":
            if reasoning_trajectories is None:
                raise ValueError("Exit calibration requires reasoning_trajectories")
            return self.compute_exit_calibration_loss(reasoning_trajectories, **objective_kwargs)

        # Drop HF-OpenVLA-only kwargs if this local PrismaticVLM wrapper is used by AVA-VLA scripts.
        kwargs.pop("proprio_projector", None)
        kwargs.pop("noisy_actions", None)
        kwargs.pop("noisy_action_projector", None)
        kwargs.pop("diffusion_timestep_embeddings", None)
        kwargs.pop("use_film", None)

        if initialize_latent_from_observation:
            if not self.enable_latent_reasoning:
                raise ValueError("Initial multimodal latent conditioning requires latent reasoning to be enabled.")
            if latent_state is not None:
                raise ValueError("Pass either latent_state or initialize_latent_from_observation, not both.")
            obs_encoding = self.encode_observation(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                history_states=history_states,
                pixel_values_wrist=pixel_values_wrist,
                proprio=proprio,
                labels=labels,
            )
            latent_state = self.initial_latent_proj(obs_encoding)

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
        wrist_image: Optional[Image.Image] = None,
        proprio: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        num_reasoning_steps: Optional[int] = None,
        return_reasoning_info: bool = False,
        history_states: Optional[torch.Tensor] = None,
        update_history: bool = True,
        center_crop: bool = False,
        **kwargs,
    ) -> np.ndarray | Tuple[np.ndarray, Dict]:
        """Predict an action; when latent reasoning is enabled, condition generation on the finalized z_t."""
        if self.enable_latent_reasoning:
            actions, info = self.predict_action_with_latent_reasoning(
                image=image,
                instruction=instruction,
                wrist_image=wrist_image,
                proprio=proprio,
                unnorm_key=unnorm_key,
                num_reasoning_steps=num_reasoning_steps,
                history_states=history_states,
                update_history=update_history,
                center_crop=center_crop,
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
        wrist_image: Optional[Image.Image] = None,
        proprio: Optional[np.ndarray] = None,
        unnorm_key: Optional[str] = None,
        num_reasoning_steps: Optional[int] = None,
        history_states: Optional[torch.Tensor] = None,
        update_history: bool = True,
        center_crop: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """Predict action and return latent-reasoning diagnostics."""
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        if center_crop:
            image = center_crop_for_augmentation(image)
            if wrist_image is not None:
                wrist_image = center_crop_for_augmentation(wrist_image)

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

        pixel_values_wrist = None
        if wrist_image is not None:
            pixel_values_wrist = image_transform(wrist_image)
            if isinstance(pixel_values_wrist, torch.Tensor):
                pixel_values_wrist = pixel_values_wrist[None, ...].to(self.device)
            elif isinstance(pixel_values_wrist, dict):
                pixel_values_wrist = {
                    key: value[None, ...].to(self.device) for key, value in pixel_values_wrist.items()
                }
            else:
                raise ValueError(f"Unsupported wrist pixel type = {type(pixel_values_wrist)}")
        proprio_tensor = None
        if proprio is not None:
            normalized_proprio = self.normalize_proprio(proprio, unnorm_key)
            proprio_tensor = torch.as_tensor(normalized_proprio, device=self.device).reshape(1, -1)

        autocast_dtype = self.llm_backbone.half_precision_dtype
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            history_states = self._resolve_history_state(history_states, input_ids.shape[0], input_ids.device)
            obs_encoding = self.encode_observation(
                pixel_values,
                input_ids,
                attention_mask=attention_mask,
                history_states=history_states,
                pixel_values_wrist=pixel_values_wrist,
                proprio=proprio_tensor,
            )
            z_0 = self.initial_latent_proj(obs_encoding)
            final_z, exit_scores, reasoning_info = self.latent_reasoning_forward(
                z_0,
                obs_encoding,
                num_steps=num_reasoning_steps,
                training=False,
                return_trajectory=False,
                force_fixed_steps=num_reasoning_steps is not None,
            )
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
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS and "min" in action_norm_stats:
            action_low = np.asarray(action_norm_stats["min"])
            action_high = np.asarray(action_norm_stats["max"])
        else:
            action_low = np.asarray(action_norm_stats["q01"])
            action_high = np.asarray(action_norm_stats["q99"])
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))
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
