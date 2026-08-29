"""Regression tests for formal AVA-VLA paper-reproduction safety contracts."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINETUNE = _load_module("finetune_avavla_safety_test", "vla-scripts/finetune_avavla.py")
DEPLOY = _load_module("deploy_avavla_safety_test", "vla-scripts/deploy_avavla.py")


class _ZeroValue:
    def __call__(self, states):
        return states.new_zeros(*states.shape[:-1], 1)


def test_semimarkov_chunk_discount() -> None:
    model = SimpleNamespace(enable_latent_reasoning=True, value_function=_ZeroValue())
    rollout = {
        "latent_states": torch.zeros(2, 1, 1),
        "next_latent_states": torch.zeros(2, 1, 1),
        "action_entropies": torch.zeros(2, 1),
        "valid_mask": torch.ones(2, 1),
        "ppo_rewards": torch.tensor([0.0, 1.0]),
        "ppo_dones": torch.tensor([False, True]),
        "ppo_chunk_lengths": torch.tensor([2, 1]),
        "ppo_env_ids": torch.tensor([0, 0]),
        "ppo_time_indices": torch.tensor([0, 1]),
        "ppo_bootstrap_values": torch.tensor([0.0]),
        "ppo_num_envs": 1,
    }
    cfg = SimpleNamespace(
        gamma=0.5,
        gae_lambda=1.0,
        entropy_coef=0.0,
        smoothness_coef=0.0,
    )
    FINETUNE.prepare_temporal_ppo_targets(model, rollout, cfg)
    assert torch.allclose(rollout["ppo_returns"].squeeze(-1), torch.tensor([0.25, 1.0]))


class _StageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(1, 1)
        self.visual_obs_proj = nn.Linear(1, 1)
        self.visual_view_fusion = nn.Linear(1, 1)
        self.text_obs_proj = nn.Linear(1, 1)
        self.proprio_obs_proj = nn.Linear(1, 1)
        self.history_obs_proj = nn.Linear(1, 1)
        self.obs_fusion = nn.Linear(1, 1)
        self.initial_latent_proj = nn.Linear(1, 1)
        self.reasoning_policy = nn.Linear(1, 1)
        self.latent_transition = nn.Linear(1, 1)
        self.value_function = nn.Linear(1, 1)
        self.latent_to_llm = nn.Linear(1, 1)
        self.exit_gate = nn.Linear(1, 1)
        self.vision_backbone = nn.Linear(1, 1)
        self.llm_backbone = nn.Linear(1, 1)
        self.latent_action_scale = nn.Parameter(torch.ones(()))


def test_stage3_joint_trainability() -> None:
    model = _StageModel()
    action_head = nn.Linear(1, 1)
    FINETUNE.configure_stage_trainability(model, action_head, "ppo", train_base_vla=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    for prefix in (
        "projector.",
        "visual_obs_proj.",
        "history_obs_proj.",
        "initial_latent_proj.",
        "reasoning_policy.",
        "latent_transition.",
        "value_function.",
        "latent_to_llm.",
    ):
        assert any(name.startswith(prefix) for name in trainable), prefix
    assert not any(name.startswith("vision_backbone.") for name in trainable)
    assert not any(name.startswith("llm_backbone.") for name in trainable)
    assert not any(name.startswith("exit_gate.") for name in trainable)
    assert all(parameter.requires_grad for parameter in action_head.parameters())


class _TinyTransition(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, update_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim + obs_dim + update_dim, latent_dim)

    def forward(self, latent, observation, update):
        return latent + self.proj(torch.cat([latent, observation, update], dim=-1))


class _TinyActionReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_fusion = nn.Linear(5, 3)
        self.initial_latent_proj = nn.Linear(3, 3)
        self.latent_transition = _TinyTransition(3, 3, 2)
        self.latent_to_llm = nn.Linear(3, 4)
        self.projector = nn.Linear(2, 4)
        self.llm_backbone = SimpleNamespace(half_precision_dtype=torch.float32)
        self.vision_backbone = SimpleNamespace(num_patches=1)

    def fuse_observation_features(self, features):
        return self.obs_fusion(
            torch.cat(
                [
                    features["primary_visual"],
                    features["wrist_visual"],
                    features["text"],
                    features["proprio"],
                    features["history"],
                ],
                dim=-1,
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        latent_state,
        output_hidden_states,
        zero_action_token_embeddings,
    ):
        del attention_mask, labels, output_hidden_states, zero_action_token_embeddings
        patch = self.projector(pixel_values.float()).unsqueeze(1)
        latent = self.latent_to_llm(latent_state).unsqueeze(1)
        token_hidden = (patch + latent).expand(-1, input_ids.shape[1], -1)
        hidden = torch.cat([patch, token_hidden], dim=1)
        return SimpleNamespace(hidden_states=[hidden])


class _TinyOFTActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(4, 1)
        with torch.no_grad():
            self.output.bias.fill_(2.0)

    def forward(self, action_features):
        batch_size = action_features.shape[0]
        grouped = action_features.reshape(
            batch_size,
            FINETUNE.NUM_ACTIONS_CHUNK,
            FINETUNE.ACTION_DIM,
            action_features.shape[-1],
        )
        return self.output(grouped).squeeze(-1)


def test_robot_action_ppo_gradient_contract() -> None:
    torch.manual_seed(41)
    batch_size, reasoning_steps = 4, 3
    model = _TinyActionReplayModel()
    action_head = _TinyOFTActionHead()
    action_count = FINETUNE.NUM_ACTIONS_CHUNK * FINETUNE.ACTION_DIM
    input_ids = torch.zeros(batch_size, action_count + 2, dtype=torch.long)
    labels = torch.full_like(input_ids, FINETUNE.IGNORE_INDEX)
    labels[:, 1 : 1 + action_count] = 31744
    labels[:, -1] = FINETUNE.STOP_INDEX
    rollout = {
        "update_actions": torch.randn(batch_size, reasoning_steps, 2),
        "valid_mask": torch.ones(batch_size, reasoning_steps),
        "ppo_action_pixel_values": torch.randn(batch_size, 2),
        "ppo_action_input_ids": input_ids,
        "ppo_action_attention_mask": torch.ones_like(input_ids),
        "ppo_action_labels": labels,
        "ppo_advantages": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 0.7]]
        ),
    }
    for name in ("primary_visual", "wrist_visual", "text", "proprio", "history"):
        rollout[f"ppo_observation_{name}"] = torch.randn(batch_size, 1)

    with torch.no_grad():
        features, _ = FINETUNE.recompute_robot_action_features(model, rollout)
        old_means = action_head(features)
        samples = old_means + 0.05 * torch.randn_like(old_means)
        old_log_probs = torch.distributions.Normal(
            old_means,
            torch.full_like(old_means, 0.05),
        ).log_prob(samples).sum(dim=(-1, -2))
    rollout["ppo_robot_action_samples"] = samples
    rollout["ppo_robot_action_means"] = old_means
    rollout["ppo_old_robot_action_log_probs"] = old_log_probs

    loss, metrics = FINETUNE.compute_robot_action_ppo_loss(
        model,
        action_head,
        rollout,
        clip_ratio=0.2,
        action_policy_std=0.05,
    )
    loss.backward()
    modules = {
        "obs_fusion": model.obs_fusion,
        "initial_latent": model.initial_latent_proj,
        "transition": model.latent_transition,
        "latent_to_llm": model.latent_to_llm,
        "projector": model.projector,
        "action_head": action_head,
    }
    for name, module in modules.items():
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        assert gradient_sum > 0.0, name
    assert torch.isfinite(loss)


def test_atomic_checkpoint_and_resume_step_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        artifact = root / "training_state--step-17-bc_checkpoint.pt"
        FINETUNE._atomic_torch_save({"log_step": 17}, artifact)
        stale = root / "training_state--step-99-ppo_checkpoint.pt"
        FINETUNE._atomic_torch_save({"log_step": 99}, stale)
        config = root / "avavla_config.json"
        FINETUNE._atomic_write_json({"implementation_version": FINETUNE.CHECKPOINT_IMPLEMENTATION_VERSION}, config)
        FINETUNE._copy_if_different(config, config)
        manifest = {
            "implementation_version": FINETUNE.CHECKPOINT_IMPLEMENTATION_VERSION,
            "log_step": 17,
            "required_files": {
                artifact.name: artifact.stat().st_size,
                config.name: config.stat().st_size,
            },
        }
        FINETUNE._atomic_write_json(manifest, root / "CHECKPOINT_COMPLETE.json")
        loaded = FINETUNE._validate_checkpoint_manifest(root)
        assert loaded["log_step"] == 17
        assert FINETUNE._find_component_checkpoint(root, "training_state", 17) == artifact
        assert FINETUNE._find_component_checkpoint(root, "training_state") == artifact
        assert DEPLOY._find_component_checkpoint(root, "training_state") == artifact
        assert "projector" in FINETUNE.AVAVLA.AVA_PARAMETER_PREFIXES

        source_base = root / "source-base.pt"
        source_base.write_bytes(b"immutable-robot-base")
        linked_base = root / "run" / "checkpoints" / "latest-checkpoint.pt"
        FINETUNE._link_or_copy_base_checkpoint(source_base, linked_base)
        assert linked_base.read_bytes() == source_base.read_bytes()
        FINETUNE._link_or_copy_base_checkpoint(source_base, linked_base)
        assert linked_base.read_bytes() == source_base.read_bytes()


def test_bc_complete_archive_survives_latest_rotation() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        artifacts = {
            "config.json": b"config",
            "dataset_statistics.json": b"stats",
            "avavla_config.json": b"ava-config",
            "checkpoints/latest-checkpoint.pt": b"immutable-base",
            "action_head--step-100000-bc_complete_checkpoint.pt": b"action-head",
            "avavla--step-100000-bc_complete_checkpoint.pt": b"ava-components",
            "training_state--step-100000-bc_complete_checkpoint.pt": b"training-state",
            "rng_state_rank0--step-100000-bc_complete_checkpoint.pt": b"rng-state",
        }
        for relative_name, payload in artifacts.items():
            path = root / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        manifest = {
            "implementation_version": FINETUNE.CHECKPOINT_IMPLEMENTATION_VERSION,
            "log_step": 100000,
            "stage": "bc_complete",
            "required_files": {
                name: (root / name).stat().st_size for name in artifacts
            },
        }
        archive = FINETUNE._archive_bc_complete_checkpoint(root, manifest)
        assert archive == root / "stage_checkpoints" / "bc_complete"
        archived_manifest = FINETUNE._validate_checkpoint_manifest(archive)
        assert archived_manifest["stage"] == "bc_complete"
        assert json.loads((archive / "STAGE_CHECKPOINT.json").read_text())["recommended_max_reasoning_steps"] == 0

        archived_action = archive / "action_head--step-100000-bc_complete_checkpoint.pt"
        assert archived_action.stat().st_ino == (
            root / "action_head--step-100000-bc_complete_checkpoint.pt"
        ).stat().st_ino
        for relative_name in artifacts:
            if Path(relative_name).name.startswith(
                ("action_head--", "avavla--", "training_state--", "rng_state_rank")
            ):
                (root / relative_name).unlink()
        assert archived_action.read_bytes() == b"action-head"
        assert FINETUNE._archive_bc_complete_checkpoint(root, manifest) == archive


def test_deployment_rejects_incomplete_or_legacy_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        try:
            DEPLOY._load_avavla_config(root)
        except RuntimeError as error:
            assert "manifest" in str(error)
        else:
            raise AssertionError("Deployment accepted a checkpoint without an atomic manifest")

        config = root / "avavla_config.json"
        config.write_text(json.dumps({"implementation_version": 2}), encoding="utf-8")
        manifest = {
            "implementation_version": DEPLOY.MIN_SAFE_IMPLEMENTATION_VERSION,
            "required_files": {config.name: config.stat().st_size},
        }
        (root / "CHECKPOINT_COMPLETE.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            DEPLOY._load_avavla_config(root)
        except RuntimeError as error:
            assert "Unsafe AVA config" in str(error)
        else:
            raise AssertionError("Deployment accepted a legacy implementation checkpoint")


def test_paper_schedule_requires_latent_reasoning() -> None:
    cfg = SimpleNamespace(enable_latent_reasoning=False)
    try:
        FINETUNE.resolve_paper_training_budget(cfg, world_size=8)
    except ValueError as error:
        assert "latent reasoning" in str(error)
    else:
        raise AssertionError("Paper schedule accepted latent reasoning disabled")


def test_ppo_checkpoint_cadence_contract() -> None:
    cfg = SimpleNamespace(
        ppo_checkpoint_interval_updates=10,
        ppo_environment_steps=1_200_000,
    )
    assert not FINETUNE.should_checkpoint_ppo(cfg, 1, 4096)
    assert not FINETUNE.should_checkpoint_ppo(cfg, 9, 9 * 4096)
    assert FINETUNE.should_checkpoint_ppo(cfg, 10, 10 * 4096)
    assert FINETUNE.should_checkpoint_ppo(cfg, 293, 1_200_000)
    defaults = FINETUNE.AVAVLAFinetuneConfig()
    assert defaults.history_window_size == FINETUNE.NUM_ACTIONS_CHUNK + 1
    assert defaults.ppo_checkpoint_interval_updates == 10
    assert defaults.action_ppo_std == 0.05
    assert defaults.action_ppo_coef == 1.0
    assert FINETUNE.CHECKPOINT_IMPLEMENTATION_VERSION >= 9


def test_openvla_oft_bidirectional_attention_patch() -> None:
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention

    source = inspect.getsource(LlamaSdpaAttention.forward)
    assert "is_causal=False" in source
    assert "last_row = causal_mask[:, :, -1, :].clone()" in source
    assert ".expand(-1, -1, D, -1)" in source


def test_robot_pretrained_openvla_base_contract() -> None:
    official, _ = FINETUNE._read_prismatic_config(PROJECT_ROOT / "models" / "openvla-7b-prismatic")
    generic, _ = FINETUNE._read_prismatic_config(
        PROJECT_ROOT / "models" / "prismatic_repo" / "prism-dinosiglip+7b"
    )
    assert official["robot_pretrained"]
    assert official["vision_backbone_id"].startswith("dinosiglip-")
    assert not generic["robot_pretrained"]



def main() -> int:
    tests = (
        test_semimarkov_chunk_discount,
        test_stage3_joint_trainability,
        test_robot_action_ppo_gradient_contract,
        test_atomic_checkpoint_and_resume_step_contract,
        test_bc_complete_archive_survives_latest_rotation,
        test_deployment_rejects_incomplete_or_legacy_checkpoint,
        test_paper_schedule_requires_latent_reasoning,
        test_ppo_checkpoint_cadence_contract,
        test_openvla_oft_bidirectional_attention_patch,
        test_robot_pretrained_openvla_base_contract,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} formal safety contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
