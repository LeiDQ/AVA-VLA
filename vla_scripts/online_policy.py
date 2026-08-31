"""Batched AVA-VLA policy queries for true online robot-environment PPO rollouts."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)
from prismatic.vla.preprocessing import center_crop_for_augmentation


def _stack_images(
    images: List[np.ndarray],
    image_transform,
    device: torch.device,
    center_crop: bool,
):
    pil_images = [Image.fromarray(image).convert("RGB") for image in images]
    if center_crop:
        pil_images = [center_crop_for_augmentation(image) for image in pil_images]
    transformed = [image_transform(image) for image in pil_images]
    if isinstance(transformed[0], torch.Tensor):
        return torch.stack(transformed).to(device)
    if isinstance(transformed[0], dict):
        return {
            key: torch.stack([row[key] for row in transformed]).to(device)
            for key in transformed[0]
        }
    raise ValueError(f"Unsupported transformed image type: {type(transformed[0])}")


def _cast_floating(value, dtype: torch.dtype):
    if isinstance(value, dict):
        return {key: _cast_floating(item, dtype) for key, item in value.items()}
    return value.to(dtype=dtype) if torch.is_floating_point(value) else value


def _prepare_policy_tokens(model, instructions: List[str], device: torch.device):
    tokenizer = model.llm_backbone.tokenizer
    prompt_rows = []
    for instruction in instructions:
        prompt_builder = model.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        row = tokenizer(
            prompt_builder.get_prompt(),
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        if isinstance(tokenizer, LlamaTokenizerFast) and row[-1].item() != 29871:
            row = torch.cat([row, row.new_tensor([29871])])
        prompt_rows.append(row)

    batch_size = len(prompt_rows)
    prompt_length = max(int(row.numel()) for row in prompt_rows)
    prompt_ids = torch.full(
        (batch_size, prompt_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    for index, row in enumerate(prompt_rows):
        length = int(row.numel())
        prompt_ids[index, :length] = row.to(device)
        prompt_mask[index, :length] = 1

    action_placeholder = torch.full(
        (batch_size, ACTION_DIM * NUM_ACTIONS_CHUNK),
        model.action_tokenizer.action_token_begin_idx + 1,
        dtype=torch.long,
        device=device,
    )
    stop_token = torch.full(
        (batch_size, 1),
        STOP_INDEX,
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.cat([prompt_ids, action_placeholder, stop_token], dim=-1)
    attention_mask = torch.cat(
        [
            prompt_mask,
            torch.ones_like(action_placeholder),
            torch.ones_like(stop_token),
        ],
        dim=-1,
    )
    labels = torch.cat(
        [
            torch.full_like(prompt_ids, IGNORE_INDEX),
            action_placeholder.clone(),
            stop_token.clone(),
        ],
        dim=-1,
    )
    return prompt_ids, prompt_mask, input_ids, attention_mask, labels


def _resolve_unnorm_keys(observations, fallback: Optional[str]) -> List[str]:
    keys: List[str] = []
    for index, observation in enumerate(observations):
        key = getattr(observation, "unnorm_key", None) or fallback
        if not key:
            raise ValueError(
                f"Online observation {index} does not declare unnorm_key and no fallback was supplied"
            )
        keys.append(str(key))
    return keys


def _normalize_proprio_batch(model, observations, unnorm_keys: Sequence[str]) -> np.ndarray:
    if len(observations) != len(unnorm_keys):
        raise ValueError("Observation and normalization-key batches must have equal length")
    rows = []
    for observation, key in zip(observations, unnorm_keys):
        proprio = np.asarray(observation.proprio, dtype=np.float32)[None]
        rows.append(np.asarray(model.normalize_proprio(proprio, key))[0])
    return np.stack(rows)


def _unnormalize_action_batch(
    model,
    actions: np.ndarray,
    unnorm_key: Union[str, Sequence[str]],
) -> np.ndarray:
    keys = [unnorm_key] * len(actions) if isinstance(unnorm_key, str) else list(unnorm_key)
    if len(keys) != len(actions):
        raise ValueError("Action and normalization-key batches must have equal length")
    output = np.empty_like(actions)
    for key in dict.fromkeys(keys):
        indices = np.asarray([index for index, value in enumerate(keys) if value == key])
        stats = model.get_action_stats(key)
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS and "min" in stats:
            low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
        else:
            low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)))
        selected = actions[indices]
        output[indices] = np.where(
            mask,
            0.5 * (selected + 1.0) * (high - low + 1e-8) + low,
            selected,
        )
    return output



_ACTION_PIXEL_PREFIX = "ppo_action_pixel_values__"


def _reconstruct_ppo_action_pixels(rollout: Dict[str, torch.Tensor]):
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
    rollout: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the canonical current z -> OpenVLA-OFT action-feature path."""
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
    input_ids = rollout["ppo_action_input_ids"]
    labels = rollout["ppo_action_labels"]
    autocast_dtype = model.llm_backbone.half_precision_dtype
    with torch.autocast(
        device_type=input_ids.device.type,
        dtype=autocast_dtype,
        enabled=input_ids.device.type == "cuda",
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
            attention_mask=rollout["ppo_action_attention_mask"],
            pixel_values=_cast_floating(
                _reconstruct_ppo_action_pixels(rollout),
                autocast_dtype,
            ),
            labels=labels,
            latent_state=current_z,
            output_hidden_states=True,
            zero_action_token_embeddings=True,
        )

    shifted_labels = labels[:, 1:]
    action_mask = get_current_action_mask(shifted_labels) | get_next_actions_mask(shifted_labels)
    text_hidden_states = output.hidden_states[-1][
        :, model.vision_backbone.num_patches : -1
    ]
    action_features = text_hidden_states[action_mask].reshape(
        input_ids.shape[0],
        NUM_ACTIONS_CHUNK * ACTION_DIM,
        -1,
    )
    return action_features, current_z
@torch.no_grad()
def query_online_policy_batch(
    avavla,
    action_head,
    observations,
    image_transform,
    unnorm_key: Optional[str],
    center_crop: bool = True,
    action_policy_std: float = 0.0,
) -> tuple[np.ndarray, Dict[str, torch.Tensor], np.ndarray]:
    """Query one online batch and retain replayable multimodal PPO features."""
    if action_policy_std < 0:
        raise ValueError("action_policy_std must be non-negative")
    model = avavla.module if hasattr(avavla, "module") else avavla
    device = model.device
    unnorm_keys = _resolve_unnorm_keys(observations, unnorm_key)
    images = _stack_images([row.image for row in observations], image_transform, device, center_crop)
    wrist_images = _stack_images(
        [row.wrist_image for row in observations], image_transform, device, center_crop
    )
    normalized_proprio = _normalize_proprio_batch(model, observations, unnorm_keys)
    proprio = torch.as_tensor(
        normalized_proprio,
        device=device,
        dtype=torch.bfloat16,
    )
    history = torch.zeros(
        len(observations),
        model.latent_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    for index, row in enumerate(observations):
        if row.history_state is not None:
            value = torch.as_tensor(row.history_state, device=device, dtype=history.dtype).flatten()
            history[index, : min(model.latent_dim, value.numel())] = value[: model.latent_dim]

    prompt_ids, prompt_mask, input_ids, attention_mask, labels = _prepare_policy_tokens(
        model,
        [row.instruction for row in observations],
        device,
    )
    autocast_enabled = device.type == "cuda"
    autocast_dtype = model.llm_backbone.half_precision_dtype
    action_pixel_values = _cast_floating(images, autocast_dtype)
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
        obs_encoding, observation_features = model.encode_observation(
            action_pixel_values,
            prompt_ids,
            attention_mask=prompt_mask,
            history_states=history,
            pixel_values_wrist=_cast_floating(wrist_images, autocast_dtype),
            proprio=proprio,
            return_features=True,
        )
        initial_z = model.initial_latent_proj(obs_encoding)
        # The cross-decision history contract is a history-free summary of the
        # current observation.  Training constructs the same summary at t-8,
        # avoiding a recursive train/inference representation mismatch.
        history_free_features = dict(observation_features)
        history_free_features["history"] = torch.zeros_like(
            observation_features["history"]
        )
        next_history_state = model.initial_latent_proj(
            model.fuse_observation_features(history_free_features)
        )
        final_z, _, trajectory = model.latent_reasoning_forward(
            initial_z,
            obs_encoding,
            num_steps=model.max_reasoning_steps,
            training=True,
            return_trajectory=True,
        )
    detached_trajectory = {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in trajectory.items()
    }
    for name, value in observation_features.items():
        detached_trajectory[f"ppo_observation_{name}"] = value.detach()
    detached_trajectory["old_action_log_probs"] = detached_trajectory["action_log_probs"].detach()
    if isinstance(action_pixel_values, dict):
        for name, value in action_pixel_values.items():
            detached_trajectory[f"ppo_action_pixel_values__{name}"] = value.detach()
    else:
        detached_trajectory["ppo_action_pixel_values"] = action_pixel_values.detach()
    detached_trajectory["ppo_action_input_ids"] = input_ids.detach()
    detached_trajectory["ppo_action_attention_mask"] = attention_mask.detach()
    detached_trajectory["ppo_action_labels"] = labels.detach()
    detached_trajectory["ppo_collected_final_latent"] = final_z.detach()

    # Collection and PPO evaluation deliberately share this exact replay path.
    # This prevents BF16 implementation drift from changing the first PPO ratio.
    action_features, _ = recompute_robot_action_features(avavla, detached_trajectory)
    action_dtype = next(action_head.parameters()).dtype
    # Match OpenVLA-OFT: the L1 head directly predicts normalized continuous
    # actions. PPO adds exploration in that same space, without a saturating
    # clamp/atanh/tanh transform.
    robot_action_means = action_head(action_features.to(action_dtype)).float()
    if action_policy_std > 0:
        action_distribution = torch.distributions.Normal(
            robot_action_means,
            torch.full_like(robot_action_means, float(action_policy_std)),
        )
        robot_action_samples = action_distribution.sample()
        old_robot_action_log_probs = action_distribution.log_prob(
            robot_action_samples
        ).sum(dim=(-1, -2))
    else:
        # Section 3.5 applies PPO to latent reasoning actions.  Keep the
        # frozen/supervised OFT action policy deterministic unless the optional
        # action-space PPO ablation is explicitly enabled.
        robot_action_samples = robot_action_means
        old_robot_action_log_probs = robot_action_means.new_zeros(robot_action_means.shape[0])
    actions = _unnormalize_action_batch(
        model,
        robot_action_samples.cpu().numpy(),
        unnorm_keys,
    )

    detached_trajectory["ppo_robot_action_samples"] = robot_action_samples.detach()
    detached_trajectory["ppo_robot_action_means"] = robot_action_means.detach()
    detached_trajectory["ppo_old_robot_action_log_probs"] = old_robot_action_log_probs.detach()
    return actions, detached_trajectory, next_history_state.float().cpu().numpy()


@torch.no_grad()
def estimate_online_bootstrap_values(
    avavla,
    collector,
    image_transform,
    unnorm_key: Optional[str],
    center_crop: bool,
) -> torch.Tensor:
    """Estimate V(z_0) for the post-rollout observations without stepping envs."""
    model = avavla.module if hasattr(avavla, "module") else avavla
    device = model.device
    observations = collector.observations()
    unnorm_keys = _resolve_unnorm_keys(observations, unnorm_key)
    images = _stack_images([row.image for row in observations], image_transform, device, center_crop)
    wrist_images = _stack_images(
        [row.wrist_image for row in observations], image_transform, device, center_crop
    )
    proprio = torch.as_tensor(
        _normalize_proprio_batch(model, observations, unnorm_keys),
        device=device,
        dtype=torch.bfloat16,
    )
    history = torch.zeros(len(observations), model.latent_dim, device=device, dtype=torch.bfloat16)
    for index, row in enumerate(observations):
        if row.history_state is not None:
            value = torch.as_tensor(row.history_state, device=device, dtype=history.dtype).flatten()
            history[index, : min(model.latent_dim, value.numel())] = value[: model.latent_dim]
    prompt_ids, prompt_mask, _, _, _ = _prepare_policy_tokens(
        model,
        [row.instruction for row in observations],
        device,
    )
    autocast_dtype = model.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        obs_encoding = model.encode_observation(
            _cast_floating(images, autocast_dtype),
            prompt_ids,
            attention_mask=prompt_mask,
            history_states=history,
            pixel_values_wrist=_cast_floating(wrist_images, autocast_dtype),
            proprio=proprio,
        )
        initial_z = model.initial_latent_proj(obs_encoding)
        values = model.value_function(initial_z).squeeze(-1)
    return values.detach()


def _merge_trajectories(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if all(torch.is_tensor(value) for value in values):
            first = values[0]
            if first.dim() > 0:
                merged[key] = torch.cat(values, dim=0)
    return merged


def collect_online_rollout(
    avavla,
    action_head,
    collector,
    image_transform,
    unnorm_key: Optional[str],
    rollout_size: int,
    center_crop: bool = True,
    action_policy_std: float = 0.0,
) -> tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
    """Collect exactly rollout_size local policy queries using true environment outcomes."""
    num_envs = len(collector.slots)
    if rollout_size % num_envs:
        raise ValueError("rollout_size must be divisible by the number of online environments")
    rows: List[Dict[str, torch.Tensor]] = []
    total_env_steps = 0
    metric_totals: Dict[str, float] = {}
    completed_episodes = 0.0
    completed_successes = 0.0
    environment_count_totals: Dict[str, float] = {}
    for rollout_time in range(rollout_size // num_envs):
        observations = collector.observations()
        actions, trajectory, final_latents = query_online_policy_batch(
            avavla,
            action_head,
            observations,
            image_transform,
            unnorm_key,
            center_crop=center_crop,
            action_policy_std=action_policy_std,
        )
        rewards, terminals, chunk_lengths, env_steps, env_metrics = collector.step_action_chunks(
            actions, final_latents
        )
        trajectory["ppo_rewards"] = torch.as_tensor(
            rewards,
            device=trajectory["latent_states"].device,
            dtype=trajectory["latent_states"].dtype,
        )
        trajectory["ppo_dones"] = torch.as_tensor(
            terminals,
            device=trajectory["latent_states"].device,
            dtype=torch.bool,
        )
        trajectory["ppo_chunk_lengths"] = torch.as_tensor(
            chunk_lengths,
            device=trajectory["latent_states"].device,
            dtype=torch.long,
        )
        if torch.any(trajectory["ppo_chunk_lengths"] <= 0):
            raise RuntimeError("Every PPO policy query must execute at least one environment step.")
        trajectory["ppo_env_ids"] = torch.arange(
            num_envs,
            device=trajectory["latent_states"].device,
            dtype=torch.long,
        )
        trajectory["ppo_time_indices"] = torch.full(
            (num_envs,),
            rollout_time,
            device=trajectory["latent_states"].device,
            dtype=torch.long,
        )
        rows.append(trajectory)
        total_env_steps += int(env_steps)
        completed_episodes += float(env_metrics["online_completed_episodes"])
        completed_successes += float(env_metrics["online_completed_successes"])
        for name, value in env_metrics.items():
            if name.endswith("_completed_episodes") or name.endswith("_completed_successes"):
                environment_count_totals[name] = environment_count_totals.get(name, 0.0) + float(value)
    metric_totals["online_completed_episodes"] = completed_episodes
    metric_totals["online_completed_successes"] = completed_successes
    metric_totals["online_success_rate"] = (
        completed_successes / completed_episodes if completed_episodes else 0.0
    )
    metric_totals["online_mean_reward"] = float(
        torch.cat([row["ppo_rewards"] for row in rows]).float().mean().cpu()
    )
    for name, value in environment_count_totals.items():
        if name in {"online_completed_episodes", "online_completed_successes"}:
            continue
        metric_totals[name] = value
    suite_prefixes = {
        name[: -len("_completed_episodes")]
        for name in environment_count_totals
        if name.startswith("online_") and name.endswith("_completed_episodes")
        and name != "online_completed_episodes"
    }
    for prefix in suite_prefixes:
        episodes = metric_totals.get(f"{prefix}_completed_episodes", 0.0)
        successes = metric_totals.get(f"{prefix}_completed_successes", 0.0)
        metric_totals[f"{prefix}_success_rate"] = successes / episodes if episodes else 0.0
    merged = _merge_trajectories(rows)
    merged["ppo_bootstrap_values"] = estimate_online_bootstrap_values(
        avavla,
        collector,
        image_transform,
        unnorm_key,
        center_crop,
    )
    merged["ppo_num_envs"] = num_envs
    return merged, total_env_steps, metric_totals
