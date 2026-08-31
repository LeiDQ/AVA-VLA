"""Fast regression checks for AVA-VLA training and inference contracts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from prismatic.models.vlas.avavla import AVAVLA, LatentTransition, ReasoningPolicy, ValueFunction
from prismatic.vla.constants import IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.vla.preprocessing import center_crop_for_augmentation


class _Vision(nn.Module):
    def forward(self, pixels):
        if isinstance(pixels, dict):
            pixels = next(iter(pixels.values()))
        pooled = pixels.float().mean(dim=tuple(range(1, pixels.dim())))
        features = torch.stack([pooled, pooled + 1, pooled + 2, pooled + 3], dim=-1)
        return features.unsqueeze(1).expand(-1, 2, -1)


class _FrozenTextModel:
    def __init__(self):
        self.last_input_ids = None

    def __call__(self, input_ids, attention_mask, **_):
        self.last_input_ids = input_ids.detach().clone()
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 5)
        return SimpleNamespace(last_hidden_state=hidden)


def test_prompt_only_observation_encoding() -> None:
    text_model = _FrozenTextModel()
    dummy = SimpleNamespace(
        enable_latent_reasoning=True,
        vision_backbone_requires_grad=False,
        vision_backbone=_Vision(),
        visual_view_fusion=nn.Linear(8, 4),
        visual_obs_proj=nn.Linear(4, 3),
        llm_backbone=SimpleNamespace(
            pad_token_id=0,
            llm=SimpleNamespace(model=text_model),
        ),
        text_obs_proj=nn.Linear(5, 3),
        proprio_dim=2,
        proprio_obs_proj=nn.Linear(2, 3),
        latent_dim=2,
        history_obs_proj=nn.Linear(2, 3),
        obs_fusion=nn.Linear(12, 3),
    )
    dummy.fuse_observation_features = lambda features: AVAVLA.fuse_observation_features(dummy, features)
    pixels = torch.ones(2, 3, 4, 4)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 300, 301],
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 999, 998],
        ]
    )
    input_ids = torch.tensor([[1, 2, 3, 300, 301], [1, 2, 3, 999, 998]])
    encoded = AVAVLA.encode_observation(
        dummy,
        pixels,
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
    )
    assert torch.allclose(encoded[0], encoded[1], atol=1e-6)
    assert text_model.last_input_ids[:, 3:].eq(0).all()


class _BatchTrackingPolicy:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, z_t, _obs):
        self.batch_sizes.append(int(z_t.shape[0]))
        return {"mean": torch.ones_like(z_t)}

    @staticmethod
    def sample_update_action(policy_output, training):
        update = policy_output["mean"]
        zeros = update.new_zeros(update.shape[0])
        return update, zeros, zeros


class _AddTransition:
    def __call__(self, z_t, _obs, update):
        # Reproduce the CUDA-autocast boundary where recurrent kernels return
        # FP32 even when the latent stream is BF16.
        return z_t.float() + update.float()


class _ThresholdExit:
    def __call__(self, z_t):
        return torch.where(z_t[:, :1] >= 1.0, z_t.new_tensor(0.9), z_t.new_tensor(0.1))


def test_dynamic_early_exit() -> None:
    policy = _BatchTrackingPolicy()
    dummy = SimpleNamespace(
        enable_latent_reasoning=True,
        max_reasoning_steps=3,
        exit_threshold=0.5,
        update_dim=1,
        reasoning_policy=policy,
        latent_transition=_AddTransition(),
        exit_gate=_ThresholdExit(),
    )
    initial = torch.tensor([[0.0], [-1.0], [-2.0]], dtype=torch.bfloat16)
    final_z, _, info = AVAVLA.latent_reasoning_forward(
        dummy,
        initial,
        torch.zeros(3, 1),
        training=False,
        return_trajectory=True,
    )
    assert policy.batch_sizes == [3, 2, 1]
    assert info["num_steps_per_sample"].tolist() == [1, 2, 3]
    assert torch.allclose(info["final_exit_scores"], torch.full((3,), 0.9))
    assert torch.allclose(final_z, torch.ones_like(final_z))
    assert final_z.dtype == torch.bfloat16


class _ZeroValue:
    def __call__(self, states):
        return states.new_zeros(*states.shape[:-1], 1)


def test_sparse_task_reward_is_not_repeated() -> None:
    steps = 4
    dummy = SimpleNamespace(enable_latent_reasoning=True, value_function=_ZeroValue())
    trajectories = {
        "latent_states": torch.zeros(1, steps, 2),
        "next_latent_states": torch.zeros(1, steps, 2),
        "action_log_probs": torch.zeros(1, steps),
        "old_action_log_probs": torch.zeros(1, steps),
        "action_entropies": torch.zeros(1, steps),
        "valid_mask": torch.ones(1, steps),
    }
    loss, metrics = AVAVLA.compute_rl_loss(
        dummy,
        trajectories,
        torch.ones(1),
        entropy_coef=0.0,
        smoothness_coef=0.0,
        recompute_policy=False,
        recompute_dynamics=False,
    )
    assert torch.isfinite(loss)
    assert abs(metrics["mean_composite_reward"] - 1.0 / steps) < 1e-6


def test_mixed_precision_ppo_replay() -> None:
    """BF16 rollout storage must replay safely through the FP32 PPO heads."""
    torch.manual_seed(11)
    batch_size, steps, latent_dim, obs_dim, update_dim = 2, 2, 8, 8, 4
    policy = ReasoningPolicy(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dim=16,
        update_dim=update_dim,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    transition = LatentTransition(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dim=16,
        update_dim=update_dim,
        num_heads=2,
        num_obs_tokens=2,
        dropout=0.0,
    )
    value = ValueFunction(latent_dim=latent_dim, hidden_dim=16, dropout=0.0)
    dummy = SimpleNamespace(
        enable_latent_reasoning=True,
        reasoning_policy=policy,
        latent_transition=transition,
        value_function=value,
    )
    trajectories = {
        "latent_states": torch.randn(batch_size, steps, latent_dim).to(torch.bfloat16),
        "next_latent_states": torch.randn(batch_size, steps, latent_dim).to(torch.bfloat16),
        "update_actions": torch.randn(batch_size, steps, update_dim).to(torch.bfloat16),
        "obs_encodings": torch.randn(batch_size, obs_dim).to(torch.bfloat16),
        "action_log_probs": torch.zeros(batch_size, steps, dtype=torch.bfloat16),
        "old_action_log_probs": torch.zeros(batch_size, steps, dtype=torch.bfloat16),
        "action_entropies": torch.zeros(batch_size, steps, dtype=torch.bfloat16),
        "valid_mask": torch.ones(batch_size, steps, dtype=torch.bfloat16),
    }
    loss, metrics = AVAVLA.compute_rl_loss(
        dummy,
        trajectories,
        torch.ones(batch_size, dtype=torch.bfloat16),
        recompute_policy=True,
        recompute_dynamics=True,
    )
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in metrics.values())
    gradients = [
        parameter.grad
        for module in (policy, transition, value)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


class _ConstantExit(nn.Module):
    def forward(self, states):
        return states.new_zeros(states.shape[0], 1)


def test_fp32_on_policy_ratio_contract() -> None:
    """An unchanged policy must replay its own rollout at ratio=1 and KL=0."""
    torch.manual_seed(17)
    latent_dim, obs_dim, update_dim = 8, 8, 4
    policy = ReasoningPolicy(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dim=16,
        update_dim=update_dim,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    ).eval()
    transition = LatentTransition(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        hidden_dim=16,
        update_dim=update_dim,
        num_heads=2,
        num_obs_tokens=2,
        dropout=0.0,
    ).eval()
    value = ValueFunction(latent_dim=latent_dim, hidden_dim=16, dropout=0.0).eval()
    dummy = SimpleNamespace(
        enable_latent_reasoning=True,
        max_reasoning_steps=3,
        exit_threshold=0.5,
        update_dim=update_dim,
        reasoning_policy=policy,
        latent_transition=transition,
        exit_gate=_ConstantExit(),
        value_function=value,
    )
    _, _, trajectory = AVAVLA.latent_reasoning_forward(
        dummy,
        torch.randn(5, latent_dim).to(torch.bfloat16),
        torch.randn(5, obs_dim).to(torch.bfloat16),
        training=True,
        return_trajectory=True,
    )
    assert trajectory["latent_states"].dtype == torch.float32
    trajectory["ppo_advantages"] = torch.ones(5, 3)
    trajectory["ppo_returns"] = torch.zeros(5, 3)
    trajectory["ppo_step_rewards"] = torch.zeros(5, 3)
    loss, metrics = AVAVLA.compute_rl_loss(
        dummy,
        trajectory,
        torch.zeros(5),
        recompute_policy=True,
        recompute_dynamics=False,
        recompute_observation=False,
    )
    assert torch.isfinite(loss)
    assert abs(metrics["ppo_ratio_mean"] - 1.0) < 1e-6
    assert metrics["ppo_clip_fraction"] == 0.0
    assert metrics["ppo_approx_kl"] < 1e-7


def test_continuous_uncertainty_penalty_cannot_reward_collapse() -> None:
    policy = ReasoningPolicy(
        latent_dim=4,
        obs_dim=4,
        hidden_dim=8,
        update_dim=3,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    mean = torch.zeros(2, 3)
    collapsed = {"mean": mean, "log_std": torch.full_like(mean, -5.0)}
    collapsed["std"] = collapsed["log_std"].exp()
    unit = {"mean": mean, "log_std": torch.zeros_like(mean)}
    unit["std"] = unit["log_std"].exp()
    collapsed_penalty = policy.uncertainty_penalty(collapsed)
    unit_penalty = policy.uncertainty_penalty(unit)
    assert torch.equal(collapsed_penalty, torch.zeros_like(collapsed_penalty))
    assert torch.all(unit_penalty > 0)


class _PPOGradientModel(nn.Module):
    """Small real-module graph used to prove PPO gradients reach the observation path."""

    def __init__(self) -> None:
        super().__init__()
        self.enable_latent_reasoning = True
        self.latent_dim = 8
        self.proprio_dim = 2
        self.visual_view_fusion = nn.Sequential(nn.Linear(8, 4), nn.GELU())
        self.visual_obs_proj = nn.Linear(4, 8)
        self.text_obs_proj = nn.Linear(5, 8)
        self.proprio_obs_proj = nn.Linear(2, 8)
        self.history_obs_proj = nn.Linear(8, 8)
        self.obs_fusion = nn.Sequential(nn.Linear(32, 8), nn.Tanh())
        self.initial_latent_proj = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.reasoning_policy = ReasoningPolicy(
            latent_dim=8,
            obs_dim=8,
            hidden_dim=16,
            update_dim=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        self.latent_transition = LatentTransition(
            latent_dim=8,
            obs_dim=8,
            hidden_dim=16,
            update_dim=4,
            num_heads=2,
            num_obs_tokens=2,
            dropout=0.0,
        )
        self.value_function = ValueFunction(latent_dim=8, hidden_dim=16, dropout=0.0)

    def fuse_observation_features(self, features):
        return AVAVLA.fuse_observation_features(self, features)


def test_ppo_observation_recompute_gradient_contract() -> None:
    torch.manual_seed(23)
    model = _PPOGradientModel()
    batch_size, steps = 3, 2
    trajectories = {
        "latent_states": torch.randn(batch_size, steps, 8),
        "next_latent_states": torch.randn(batch_size, steps, 8),
        "update_actions": torch.randn(batch_size, steps, 4),
        "obs_encodings": torch.randn(batch_size, 8),
        "action_log_probs": torch.zeros(batch_size, steps),
        "old_action_log_probs": torch.zeros(batch_size, steps),
        "action_entropies": torch.zeros(batch_size, steps),
        "valid_mask": torch.ones(batch_size, steps),
        "ppo_advantages": torch.tensor([[1.0, 0.5], [0.7, 0.2], [1.2, 0.3]]),
        "ppo_returns": torch.randn(batch_size, steps),
        "ppo_step_rewards": torch.zeros(batch_size, steps),
        "ppo_observation_primary_visual": torch.randn(batch_size, 4),
        "ppo_observation_wrist_visual": torch.randn(batch_size, 4),
        "ppo_observation_text": torch.randn(batch_size, 5),
        "ppo_observation_proprio": torch.randn(batch_size, 2),
        "ppo_observation_history": torch.randn(batch_size, 8),
    }
    loss, metrics = AVAVLA.compute_rl_loss(
        model,
        trajectories,
        torch.zeros(batch_size),
        recompute_policy=True,
        recompute_dynamics=True,
        recompute_observation=True,
    )
    loss.backward()
    modules = {
        "visual_view_fusion": model.visual_view_fusion,
        "visual_obs_proj": model.visual_obs_proj,
        "text_obs_proj": model.text_obs_proj,
        "proprio_obs_proj": model.proprio_obs_proj,
        "history_obs_proj": model.history_obs_proj,
        "obs_fusion": model.obs_fusion,
        "initial_latent_proj": model.initial_latent_proj,
        "reasoning_policy": model.reasoning_policy,
        "latent_transition": model.latent_transition,
        "value_function": model.value_function,
    }
    for name, module in modules.items():
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert np.isfinite(gradient_sum) and gradient_sum > 0.0, name
    assert metrics["ppo_observation_recomputed"] == 1.0
    assert metrics["ppo_policy_recomputed"] == 1.0
    assert metrics["ppo_dynamics_recomputed"] == 1.0


def test_regularizers_are_not_double_counted() -> None:
    dummy = SimpleNamespace(enable_latent_reasoning=True, value_function=_ZeroValue())

    def make_trajectory(scale: float):
        return {
            "latent_states": torch.zeros(1, 2, 2),
            "next_latent_states": torch.full((1, 2, 2), scale),
            "action_log_probs": torch.zeros(1, 2),
            "old_action_log_probs": torch.zeros(1, 2),
            "action_entropies": torch.full((1, 2), scale),
            "valid_mask": torch.ones(1, 2),
            "ppo_advantages": torch.ones(1, 2),
            "ppo_returns": torch.zeros(1, 2),
            "ppo_step_rewards": torch.zeros(1, 2),
        }

    low_loss, low_metrics = AVAVLA.compute_rl_loss(
        dummy,
        make_trajectory(1.0),
        torch.zeros(1),
        entropy_coef=0.01,
        smoothness_coef=0.1,
        recompute_policy=False,
        recompute_dynamics=False,
        recompute_observation=False,
    )
    high_loss, high_metrics = AVAVLA.compute_rl_loss(
        dummy,
        make_trajectory(100.0),
        torch.zeros(1),
        entropy_coef=0.01,
        smoothness_coef=0.1,
        recompute_policy=False,
        recompute_dynamics=False,
        recompute_observation=False,
    )
    assert torch.allclose(low_loss, high_loss)
    assert high_metrics["entropy_penalty"] > low_metrics["entropy_penalty"]
    assert high_metrics["smoothness_loss"] > low_metrics["smoothness_loss"]
    assert high_metrics["regularizers_in_reward_only"] == 1.0


def test_proprio_mask_and_center_crop() -> None:
    stats = {
        "suite": {
            "proprio": {
                "min": [0.0, -1.0],
                "max": [2.0, 1.0],
                "q01": [0.0, -1.0],
                "q99": [2.0, 1.0],
                "mask": [True, False],
            }
        }
    }
    dummy = SimpleNamespace(norm_stats=stats)
    normalized = AVAVLA.normalize_proprio(dummy, np.array([2.0, 3.0]), "suite")
    assert np.allclose(normalized, np.array([1.0, 3.0], dtype=np.float32))

    image = Image.fromarray(np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3))
    cropped_a = center_crop_for_augmentation(image)
    cropped_b = center_crop_for_augmentation(image)
    assert cropped_a.size == image.size
    assert np.array_equal(np.asarray(cropped_a), np.asarray(cropped_b))


def test_rlds_action_and_history_contract() -> None:
    from transformers import LlamaTokenizerFast

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform

    tokenizer_dir = PROJECT_ROOT / "models" / "llama2-7b-ms-tokenizer"
    assert tokenizer_dir.is_dir(), f"Missing smoke-test tokenizer: {tokenizer_dir}"
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_dir)
    action_tokenizer = ActionTokenizer(tokenizer)

    class _ImageTransform:
        def __call__(self, image):
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
            return {"dino": tensor, "siglip": tensor.clone()}

    transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=_ImageTransform(),
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    actions = np.linspace(-1.0, 1.0, 16 * 7, dtype=np.float32).reshape(16, 7)
    history_length = NUM_ACTIONS_CHUNK + 1
    primary_images = np.stack(
        [np.full((16, 16, 3), index, dtype=np.uint8) for index in range(history_length)]
    )
    wrist_images = np.stack(
        [np.full((16, 16, 3), index + 20, dtype=np.uint8) for index in range(history_length)]
    )
    proprio = np.arange(history_length * 8, dtype=np.float32).reshape(history_length, 8)
    rlds_batch = {
        "dataset_name": b"libero_spatial_no_noops",
        "action": actions,
        "observation": {
            "image_primary": primary_images,
            "image_wrist": wrist_images,
            "proprio": proprio,
            "pad_mask": np.array([False] + [True] * NUM_ACTIONS_CHUNK),
        },
        "task": {"language_instruction": b"put the object in the bowl"},
    }
    instance = transform(rlds_batch)
    assert instance["actions"].shape == (8, 7)
    assert np.allclose(instance["actions"], actions[-NUM_ACTIONS_CHUNK:])
    shifted_labels = instance["labels"][1:].unsqueeze(0)
    action_mask = get_current_action_mask(shifted_labels) | get_next_actions_mask(shifted_labels)
    assert int(action_mask.sum()) == 8 * 7
    assert "pixel_values_history" in instance
    assert "pixel_values_wrist_history" in instance
    assert not bool(instance["history_pad_mask"])
    assert np.allclose(instance["proprio_history"], proprio[0])
    assert np.allclose(instance["proprio"], proprio[-1])
    assert float(instance["pixel_values_history"]["dino"].mean()) == 0.0
    assert float(instance["pixel_values"]["dino"].mean()) > 0.0

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
    )
    batch = collator([instance, instance])
    assert batch["actions"].shape == (2, 8, 7)
    assert batch["proprio"].shape == (2, 8)
    assert batch["pixel_values"]["dino"].shape[:2] == (2, 3)
    assert batch["pixel_values_wrist"]["dino"].shape[:2] == (2, 3)
    assert batch["pixel_values_history"]["dino"].shape[:2] == (2, 3)
    assert batch["history_pad_mask"].tolist() == [False, False]


def main() -> int:
    tests = [
        test_prompt_only_observation_encoding,
        test_dynamic_early_exit,
        test_sparse_task_reward_is_not_repeated,
        test_mixed_precision_ppo_replay,
        test_fp32_on_policy_ratio_contract,
        test_continuous_uncertainty_penalty_cannot_reward_collapse,
        test_ppo_observation_recompute_gradient_contract,
        test_regularizers_are_not_double_counted,
        test_proprio_mask_and_center_crop,
        test_rlds_action_and_history_contract,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} AVA-VLA regression contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
