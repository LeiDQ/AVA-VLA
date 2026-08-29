#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${1:?usage: run_official_base_preflight.sh BASE_MODEL}"
PYTHON="$ROOT/.venv/bin/python"
RUN_ROOT="$ROOT/runs/preflight"
RUN_ID="official_openvla_avavla_preflight"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_ROOT="$ROOT/logs/paper_reproduction"
TRAIN_LOG="$LOG_ROOT/preflight.train.log"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

validate_run() {
    "$PYTHON" - "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "CHECKPOINT_COMPLETE.json").open() as stream:
    manifest = json.load(stream)
with (root / "TRAINING_COMPLETE").open() as stream:
    complete = json.load(stream)
if int(manifest.get("implementation_version", 0)) < 9:
    raise SystemExit("checkpoint schema is incompatible with the current action-PPO contract")
for relative_name, expected_size in manifest.get("required_files", {}).items():
    path = root / relative_name
    if not path.is_file() or path.stat().st_size != int(expected_size):
        raise SystemExit(f"missing checkpoint artifact: {relative_name}")
if int(complete.get("ppo_environment_steps", 0)) < 64:
    raise SystemExit("preflight did not execute enough real LIBERO environment steps")
rows = []
with (root / "metrics.jsonl").open() as stream:
    for line in stream:
        rows.append(json.loads(line))
if not rows:
    raise SystemExit("preflight metrics are empty")
steps = [int(row["global_step"]) for row in rows]
if steps != list(range(steps[0], steps[-1] + 1)) or steps[0] != 1:
    raise SystemExit(f"non-contiguous global steps: {steps}")
required = {
    "bc": {"action_loss", "curr_action_l1_loss", "next_actions_l1_loss", "total_loss"},
    "latent_warmup": {"latent_warmup_loss", "latent_distance_mean"},
    "online_ppo": {
        "policy_loss", "value_loss", "smoothness_loss", "total_rl_loss",
        "robot_action_policy_loss", "joint_ppo_loss", "ppo_observation_recomputed",
        "ppo_joint_action_update", "ppo_action_path_recomputed", "ppo_bc_action_loss",
        "ppo_projector_grad_norm", "ppo_latent_to_llm_grad_norm",
        "ppo_action_head_grad_norm", "robot_action_mean_out_of_bounds_fraction",
    },
    "exit_calibration": {"exit_calibration_loss", "exit_calibration_accuracy"},
}
seen = set()
for row in rows:
    stage = row.get("stage")
    metrics = row.get("metrics")
    if stage not in required or not isinstance(metrics, dict):
        raise SystemExit(f"invalid metric row: {row}")
    if not required[stage].issubset(metrics):
        raise SystemExit(f"stage {stage} misses {sorted(required[stage] - set(metrics))}")
    values = list(metrics.values()) + [row.get("grad_norm")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        raise SystemExit(f"stage {stage} contains a non-finite metric")
    seen.add(stage)
if seen != set(required):
    raise SystemExit(f"missing preflight stages: {sorted(set(required) - seen)}")
print(f"valid_preflight rows={len(rows)} final_step={steps[-1]} env_steps={complete['ppo_environment_steps']}")
PY
}

if [[ -d "$RUN_DIR" ]] && validate_run "$RUN_DIR" >/dev/null 2>&1; then
    printf '%s preflight_reused run=%s\n' "$(date -Is)" "$RUN_ID"
    exit 0
fi
if [[ -d "$RUN_DIR" ]]; then
    archive="$RUN_ROOT/invalid/$RUN_ID.$(date -u +%Y%m%dT%H%M%S%N)"
    mkdir -p "$(dirname "$archive")"
    mv "$RUN_DIR" "$archive"
    printf '%s preflight_archived archive=%s\n' "$(date -Is)" "$archive"
fi

printf '%s preflight_training_start base=%s\n' "$(date -Is)" "$BASE_MODEL"
"$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
    "$ROOT/vla-scripts/finetune_avavla.py" \
    --vla_path "$BASE_MODEL" \
    --llm_config_path "$ROOT/models/llama2-7b-ms-tokenizer" \
    --data_root_dir "$ROOT/data/modified_libero_rlds" \
    --dataset_name libero_spatial_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --run_id_override "$RUN_ID" \
    --seed 314159 \
    --batch_size 1 \
    --grad_accumulation_steps 1 \
    --shuffle_buffer_size 64 \
    --image_aug false \
    --use_wrist_image true \
    --use_proprio true \
    --require_384px_backbone false \
    --require_dinosiglip_backbone true \
    --history_window_size 9 \
    --train_base_vla false \
    --bc_steps 1 \
    --latent_warmup_steps 1 \
    --ppo_effective_batch_size 8 \
    --ppo_rollout_size_per_rank 1 \
    --online_envs_per_rank 1 \
    --online_center_crop true \
    --ppo_environment_steps 64 \
    --ppo_minibatch_size 8 \
    --ppo_epochs 1 \
    --policy_lr 3e-5 \
    --critic_lr 1e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --ppo_clip_ratio 0.2 \
    --entropy_coef 0.01 \
    --smoothness_coef 0.1 \
    --action_ppo_std 0.05 \
    --action_ppo_coef 1.0 \
    --max_grad_norm 1.0 \
    --max_reasoning_steps 5 \
    --exit_threshold 0.55 \
    --exit_calibration_steps 1 \
    --exit_calibration_buffer_rollouts 2 \
    --online_num_steps_wait 1 \
    --libero_task_suites libero_spatial \
    --require_online_task_rewards true \
    --save_freq 1 \
    --ppo_checkpoint_interval_updates 1 \
    --exit_checkpoint_interval_steps 1 \
    --save_latest_checkpoint_only true \
    --use_wandb false >>"$TRAIN_LOG" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
    printf '%s preflight_training_failed exit_code=%s log=%s\n' "$(date -Is)" "$rc" "$TRAIN_LOG" >&2
    exit "$rc"
fi
validate_run "$RUN_DIR"
printf '%s preflight_complete run=%s\n' "$(date -Is)" "$RUN_ID"
