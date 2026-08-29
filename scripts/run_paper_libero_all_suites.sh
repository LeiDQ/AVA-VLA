#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISION_BACKBONE_PROFILE="${VISION_BACKBONE_PROFILE:-openvla-224}"
case "$VISION_BACKBONE_PROFILE" in
    openvla-224)
        DEFAULT_RUN_ROOT="$ROOT/runs/paper_all_suites"
        DEFAULT_LOG_ROOT="$ROOT/logs/paper_all_suites"
        DEFAULT_RESULT_ROOT="$ROOT/results/paper_all_suites"
        REQUIRE_384PX_BACKBONE=false
        ;;
    dinosiglip-384)
        DEFAULT_RUN_ROOT="$ROOT/runs/paper_all_suites_dinosiglip384"
        DEFAULT_LOG_ROOT="$ROOT/logs/paper_all_suites_dinosiglip384"
        DEFAULT_RESULT_ROOT="$ROOT/results/paper_all_suites_dinosiglip384"
        REQUIRE_384PX_BACKBONE=true
        ;;
    *)
        printf 'Unsupported VISION_BACKBONE_PROFILE=%s (expected openvla-224 or dinosiglip-384)\n' \
            "$VISION_BACKBONE_PROFILE" >&2
        exit 2
        ;;
esac

RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
RESULT_ROOT="${RESULT_ROOT:-$DEFAULT_RESULT_ROOT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/modified_libero_rlds}"
TOKENIZER="${TOKENIZER:-$ROOT/models/llama2-7b-ms-tokenizer}"
PAPER_SEEDS="${PAPER_SEEDS:-0 1 2}"
EVAL_SHARDS="${EVAL_SHARDS:-8}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2048}"
BC_SAVE_FREQ="${BC_SAVE_FREQ:-1000}"
PYTHON="$ROOT/.venv/bin/python"
DATASET="libero_4_task_suites_no_noops"
SUITES="libero_spatial,libero_object,libero_goal,libero_10"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
exec 9>"$LOG_ROOT/pipeline.lock"
flock -n 9 || {
    echo "Another all-suite reproduction pipeline owns $LOG_ROOT/pipeline.lock" >&2
    exit 1
}
printf '%s\n' "$$" >"$LOG_ROOT/pipeline.pid"

export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/status.log"
}

for integer_setting in EVAL_SHARDS SHUFFLE_BUFFER_SIZE BC_SAVE_FREQ; do
    integer_value="${!integer_setting}"
    if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer (received %s)\n' \
            "$integer_setting" "$integer_value" >&2
        exit 2
    fi
done
if (( EVAL_SHARDS > 8 )); then
    printf 'EVAL_SHARDS must be in 1..8 (received %s)\n' "$EVAL_SHARDS" >&2
    exit 2
fi

if [[ "$VISION_BACKBONE_PROFILE" == "dinosiglip-384" && -z "${BASE_MODEL:-}" ]]; then
    printf '%s\n' \
        'VISION_BACKBONE_PROFILE=dinosiglip-384 requires BASE_MODEL to point to a verified 384px OXE robot base.' >&2
    exit 2
fi
if [[ "$VISION_BACKBONE_PROFILE" == "openvla-224" && -z "${BASE_MODEL:-}" ]]; then
    for candidate in \
        "$ROOT/models/openvla-7b-modelscope-prismatic" \
        "$ROOT/models/openvla-7b-prismatic"; do
        if "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$candidate" >/dev/null 2>&1; then
            BASE_MODEL="$candidate"
            break
        fi
    done
fi
BASE_MODEL="${BASE_MODEL:-$ROOT/models/openvla-7b-prismatic}"

if [[ "$VISION_BACKBONE_PROFILE" == "dinosiglip-384" ]]; then
    "$PYTHON" "$ROOT/scripts/validate_avavla_vision_base.py" \
        "$BASE_MODEL" --expected-resolution 384 --full-hash >/dev/null
else
    "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" \
        "$BASE_MODEL" --full-hash >/dev/null
fi
status "base_validated path=$BASE_MODEL vision_profile=$VISION_BACKBONE_PROFILE shuffle_buffer_per_rank=$SHUFFLE_BUFFER_SIZE bc_save_freq=$BC_SAVE_FREQ"
for dataset in \
    libero_spatial_no_noops \
    libero_object_no_noops \
    libero_goal_no_noops \
    libero_10_no_noops; do
    if [[ ! -d "$DATA_ROOT/$dataset/1.0.0" ]]; then
        status "blocked_missing_dataset dataset=$dataset path=$DATA_ROOT/$dataset/1.0.0"
        exit 2
    fi
done

for seed in $PAPER_SEEDS; do
    run_id="paper_all_suites_seed${seed}"
    run_dir="$RUN_ROOT/$run_id"
    train_log="$LOG_ROOT/${run_id}.train.log"
    resume_args=(--vla_path "$BASE_MODEL")
    if [[ -f "$run_dir/CHECKPOINT_COMPLETE.json" && ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
        resume_args=(--vla_path "$run_dir" --resume true)
    elif [[ -d "$run_dir" && ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
        status "blocked_unsafe_partial_run run=$run_id path=$run_dir"
        exit 3
    fi

    if [[ ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
        status "training_start policy=all_four_suites seed=$seed dataset=$DATASET"
        "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
            "$ROOT/vla-scripts/finetune_avavla.py" \
            "${resume_args[@]}" \
            --llm_config_path "$TOKENIZER" \
            --data_root_dir "$DATA_ROOT" \
            --dataset_name "$DATASET" \
            --run_root_dir "$RUN_ROOT" \
            --run_id_override "$run_id" \
            --seed "$seed" \
            --batch_size 8 \
            --grad_accumulation_steps 1 \
            --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
            --image_aug true \
            --use_wrist_image true \
            --use_proprio true \
            --require_384px_backbone "$REQUIRE_384PX_BACKBONE" \
            --require_dinosiglip_backbone true \
            --history_window_size 9 \
            --train_base_vla false \
            --bc_steps 100000 \
            --latent_warmup_steps 50000 \
            --ppo_effective_batch_size 512 \
            --ppo_rollout_size_per_rank 64 \
            --online_envs_per_rank 8 \
            --online_center_crop true \
            --ppo_environment_steps 1200000 \
            --ppo_minibatch_size 64 \
            --ppo_epochs 4 \
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
            --exit_calibration_steps 10000 \
            --exit_calibration_lookahead 3 \
            --exit_calibration_delta 0.05 \
            --online_num_steps_wait 10 \
            --libero_task_suites "$SUITES" \
            --require_online_task_rewards true \
            --save_freq "$BC_SAVE_FREQ" \
            --ppo_checkpoint_interval_updates 10 \
            --exit_checkpoint_interval_steps 1000 \
            --save_latest_checkpoint_only true \
            --use_wandb false >>"$train_log" 2>&1
    fi
    if [[ ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
        status "training_failed_without_completion_marker run=$run_id"
        exit 4
    fi

    "$PYTHON" "$ROOT/scripts/validate_avavla_stage_checkpoints.py" \
        "$run_dir" --write-report >>"$train_log" 2>&1
    status "stage_checkpoints_validated policy=all_four_suites seed=$seed run=$run_id"

    status "evaluation_start policy=all_four_suites seed=$seed checkpoint=$run_dir"
    "$PYTHON" "$ROOT/scripts/evaluate_libero_checkpoint_all_suites.py" \
        --checkpoint "$run_dir" \
        --output-root "$RESULT_ROOT/seed${seed}" \
        --shards "$EVAL_SHARDS" \
        --num-trials-per-task 50 >>"$LOG_ROOT/${run_id}.eval.log" 2>&1
done

"$PYTHON" "$ROOT/scripts/aggregate_table1_ours.py" \
    --result-root "$RESULT_ROOT" \
    --mode all_policy \
    --seeds $PAPER_SEEDS \
    --output "$RESULT_ROOT/table1_ours_all_policy.json" \
    >>"$LOG_ROOT/aggregate.log" 2>&1
status "pipeline_complete policy=all_four_suites seeds=[$PAPER_SEEDS]"
