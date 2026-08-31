#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:?usage: run_libero_suite_smoke.sh SUITE DATASET GPU_LIST [BASE_MODEL] [WORLD_SIZE] [PROFILE]}"
DATASET="${2:?usage: run_libero_suite_smoke.sh SUITE DATASET GPU_LIST [BASE_MODEL] [WORLD_SIZE] [PROFILE]}"
GPU_LIST="${3:?usage: run_libero_suite_smoke.sh SUITE DATASET GPU_LIST [BASE_MODEL] [WORLD_SIZE] [PROFILE]}"
BASE_MODEL="${4:-$ROOT/models/openvla-7b-modelscope-prismatic}"
WORLD_SIZE="${5:-1}"
PROFILE="${6:-5min}"
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'WORLD_SIZE must be a positive integer, got %s\n' "$WORLD_SIZE" >&2
    exit 2
fi
case "$PROFILE" in
    5min)
        BC_STEPS=80
        LATENT_WARMUP_STEPS=30
        MIN_PPO_ROWS=4
        PPO_ENVIRONMENT_STEPS=$((32 * WORLD_SIZE))
        EXIT_CALIBRATION_STEPS=20
        SAVE_FREQ=20
        ;;
    100bc)
        BC_STEPS=100
        LATENT_WARMUP_STEPS=50
        MIN_PPO_ROWS=4
        PPO_ENVIRONMENT_STEPS=$((32 * WORLD_SIZE))
        EXIT_CALIBRATION_STEPS=20
        SAVE_FREQ=25
        ;;
    object_curve)
        BC_STEPS=400
        LATENT_WARMUP_STEPS=100
        MIN_PPO_ROWS=4
        PPO_ENVIRONMENT_STEPS=$((32 * WORLD_SIZE))
        EXIT_CALIBRATION_STEPS=20
        SAVE_FREQ=100
        ;;
    *)
        printf 'Unsupported smoke profile: %s\n' "$PROFILE" >&2
        exit 2
        ;;
esac
PYTHON="$ROOT/.venv/bin/python"
RUN_ROOT="$ROOT/runs/suite_smoke"
RUN_ID="official_openvla_smoke_${SUITE}_${PROFILE}_${WORLD_SIZE}gpu"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG_ROOT="$ROOT/logs/suite_smoke/${SUITE}_${PROFILE}_${WORLD_SIZE}gpu"
TRAIN_LOG="$LOG_ROOT/train.log"
STATUS_LOG="$LOG_ROOT/status.log"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/supervisor.pid"
export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_LOG"
}

validate_run() {
    "$PYTHON" - "$1" "$PPO_ENVIRONMENT_STEPS" "$BC_STEPS" \
        "$LATENT_WARMUP_STEPS" "$MIN_PPO_ROWS" "$EXIT_CALIBRATION_STEPS" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
minimum_environment_steps = int(sys.argv[2])
minimum_stage_rows = {
    "bc": int(sys.argv[3]),
    "latent_warmup": int(sys.argv[4]),
    "online_ppo": int(sys.argv[5]),
    "exit_calibration": int(sys.argv[6]),
}
with (root / "CHECKPOINT_COMPLETE.json").open() as stream:
    manifest = json.load(stream)
with (root / "TRAINING_COMPLETE").open() as stream:
    complete = json.load(stream)
if int(manifest.get("implementation_version", 0)) < 9:
    raise SystemExit("checkpoint schema is incompatible with the current PPO contract")
for relative_name, expected_size in manifest.get("required_files", {}).items():
    path = root / relative_name
    if not path.is_file() or path.stat().st_size != int(expected_size):
        raise SystemExit(f"missing checkpoint artifact: {relative_name}")
if int(complete.get("ppo_environment_steps", 0)) < minimum_environment_steps:
    raise SystemExit("smoke did not execute enough real LIBERO environment steps")
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line]
if not rows:
    raise SystemExit("smoke metrics are empty")
steps = [int(row["global_step"]) for row in rows]
if steps != list(range(1, steps[-1] + 1)):
    raise SystemExit(f"non-contiguous global steps: {steps}")
required = {
    "bc": {"action_loss", "curr_action_l1_loss", "next_actions_l1_loss", "total_loss"},
    "latent_warmup": {"latent_warmup_loss", "latent_distance_mean"},
    "online_ppo": {
        "policy_loss", "value_loss", "smoothness_loss", "total_rl_loss",
        "joint_ppo_loss", "ppo_ratio_mean", "ppo_clip_fraction", "ppo_approx_kl",
        "ppo_global_max_kl", "ppo_early_stop_kl", "ppo_optimizer_minibatches",
        "ppo_pre_update_approx_kl", "ppo_trust_region_scale", "ppo_trust_region_backtracks",
        "robot_action_ppo_enabled", "ppo_bc_action_loss",
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
    if stage == "online_ppo":
        if float(metrics["ppo_global_max_kl"]) > 0.030001:
            raise SystemExit(f"PPO trust region was violated: {metrics['ppo_global_max_kl']}")
        if float(metrics["ppo_trust_region_scale"]) <= 0.0:
            raise SystemExit("PPO candidate update could not be accepted")
if seen != set(required):
    raise SystemExit(f"missing smoke stages: {sorted(set(required) - seen)}")
stage_counts = Counter(row["stage"] for row in rows)
for stage, minimum in minimum_stage_rows.items():
    if stage_counts[stage] < minimum:
        raise SystemExit(f"stage {stage} has {stage_counts[stage]} rows, expected at least {minimum}")
print(
    f"valid_smoke suite={complete['dataset']} rows={len(rows)} "
    f"env_steps={complete['ppo_environment_steps']} stage_counts={dict(stage_counts)}"
)
PY
}

plot_run() {
    "$PYTHON" "$ROOT/scripts/plot_avavla_losses.py" \
        "$RUN_DIR/metrics.jsonl" \
        --output-dir "$RUN_DIR/training_curves" \
        --smoothing 0.1 >>"$TRAIN_LOG" 2>&1
}

attempt=0
while true; do
    if [[ -d "$RUN_DIR" ]] && validate_run "$RUN_DIR" >>"$TRAIN_LOG" 2>&1; then
        if plot_run; then
            status "complete suite=$SUITE run=$RUN_ID reused=true curves=$RUN_DIR/training_curves"
            exit 0
        fi
        status "curve_generation_failed suite=$SUITE retry_seconds=30"
        sleep 30
        continue
    fi
    if [[ -d "$RUN_DIR" ]]; then
        archive="$RUN_ROOT/invalid/$RUN_ID.$(date -u +%Y%m%dT%H%M%S%N)"
        mkdir -p "$(dirname "$archive")"
        mv "$RUN_DIR" "$archive"
        status "archived suite=$SUITE archive=$archive"
    fi

    attempt=$((attempt + 1))
    status "start suite=$SUITE dataset=$DATASET physical_gpus=$GPU_LIST world_size=$WORLD_SIZE profile=$PROFILE bc=$BC_STEPS warmup=$LATENT_WARMUP_STEPS ppo_env=$PPO_ENVIRONMENT_STEPS gate=$EXIT_CALIBRATION_STEPS attempt=$attempt"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node="$WORLD_SIZE" \
        "$ROOT/vla-scripts/finetune_avavla.py" \
        --vla_path "$BASE_MODEL" \
        --llm_config_path "$ROOT/models/llama2-7b-ms-tokenizer" \
        --data_root_dir "$ROOT/data/modified_libero_rlds" \
        --dataset_name "$DATASET" \
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
        --bc_steps "$BC_STEPS" \
        --latent_warmup_steps "$LATENT_WARMUP_STEPS" \
        --ppo_effective_batch_size "$WORLD_SIZE" \
        --ppo_rollout_size_per_rank 1 \
        --online_envs_per_rank 1 \
        --online_center_crop true \
        --ppo_environment_steps "$PPO_ENVIRONMENT_STEPS" \
        --ppo_minibatch_size "$WORLD_SIZE" \
        --ppo_epochs 1 \
        --policy_lr 3e-5 \
        --critic_lr 1e-4 \
        --gamma 0.99 \
        --gae_lambda 0.95 \
        --ppo_clip_ratio 0.2 \
        --entropy_coef 0.01 \
        --smoothness_coef 0.1 \
        --action_ppo_std 0.05 \
        --action_ppo_coef 0.0 \
        --ppo_target_kl 0.02 \
        --ppo_max_backtracks 12 \
        --max_grad_norm 1.0 \
        --max_reasoning_steps 5 \
        --exit_threshold 0.55 \
        --exit_calibration_steps "$EXIT_CALIBRATION_STEPS" \
        --exit_calibration_buffer_rollouts 4 \
        --online_num_steps_wait 1 \
        --libero_task_suites "$SUITE" \
        --require_online_task_rewards true \
        --save_freq "$SAVE_FREQ" \
        --ppo_checkpoint_interval_updates 1 \
        --exit_checkpoint_interval_steps 10 \
        --save_latest_checkpoint_only true \
        --use_wandb false >>"$TRAIN_LOG" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]] && validate_run "$RUN_DIR" >>"$TRAIN_LOG" 2>&1; then
        if plot_run; then
            status "complete suite=$SUITE run=$RUN_ID reused=false curves=$RUN_DIR/training_curves"
            exit 0
        fi
        status "curve_generation_failed suite=$SUITE retry_seconds=30"
        sleep 30
        continue
    fi
    status "failed suite=$SUITE exit_code=$rc retry_seconds=30 log=$TRAIN_LOG"
    sleep 30
done
