#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${CALVIN_DATA_ROOT:-$ROOT/calvin_data/task_ABC_D}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/paper_calvin}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/paper_calvin}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/paper_calvin}"
TOKENIZER="${TOKENIZER:-$ROOT/models/llama2-7b-ms-tokenizer}"
BASE_MODEL="${BASE_MODEL:-}"
SEED="${SEED:-0}"
EVAL_SHARDS="${EVAL_SHARDS:-8}"
PYTHON="$ROOT/.venv/bin/python"

if [[ -z "$BASE_MODEL" ]]; then
    for candidate in "$ROOT/models/openvla-7b-modelscope-prismatic" "$ROOT/models/openvla-7b-prismatic"; do
        if "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$candidate" >/dev/null 2>&1; then
            BASE_MODEL="$candidate"
            break
        fi
    done
fi
if [[ -z "$BASE_MODEL" ]]; then
    echo "No valid OpenVLA Prismatic base checkpoint found" >&2
    exit 2
fi
if [[ ! -f "$DATA_ROOT/training/lang_annotations/auto_lang_ann.npy" || \
      ! -f "$DATA_ROOT/validation/.hydra/merged_config.yaml" ]]; then
    echo "CALVIN ABC->D data is incomplete at $DATA_ROOT" >&2
    exit 3
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT/third_party/calvin_runtime:$ROOT/third_party/calvin/calvin_models:$ROOT/third_party/calvin/calvin_env:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

RUN_ID="paper_calvin_abc_seed${SEED}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
TRAIN_LOG="$LOG_ROOT/${RUN_ID}.train.log"
resume_args=(--vla_path "$BASE_MODEL")
if [[ -f "$RUN_DIR/CHECKPOINT_COMPLETE.json" ]]; then
    resume_args=(--vla_path "$RUN_DIR" --resume true)
fi

if [[ ! -f "$RUN_DIR/TRAINING_COMPLETE" ]]; then
    "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
        "$ROOT/vla-scripts/finetune_avavla.py" \
        "${resume_args[@]}" \
        --llm_config_path "$TOKENIZER" \
        --data_root_dir "$DATA_ROOT" \
        --dataset_name calvin_abc \
        --run_root_dir "$RUN_ROOT" \
        --run_id_override "$RUN_ID" \
        --seed "$SEED" \
        --batch_size 8 \
        --grad_accumulation_steps 1 \
        --image_aug true \
        --use_wrist_image true \
        --use_proprio true \
        --history_window_size 9 \
        --train_base_vla false \
        --bc_steps 100000 \
        --latent_warmup_steps 50000 \
        --ppo_effective_batch_size 512 \
        --ppo_rollout_size_per_rank 64 \
        --online_envs_per_rank 8 \
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
        --max_reasoning_steps 5 \
        --exit_threshold 0.55 \
        --exit_calibration_steps 10000 \
        --require_online_task_rewards true \
        --save_freq 10000 \
        --ppo_checkpoint_interval_updates 10 \
        --exit_checkpoint_interval_steps 1000 \
        --save_latest_checkpoint_only true \
        --use_wandb false >>"$TRAIN_LOG" 2>&1
fi

shard_dir="$RESULT_ROOT/$RUN_ID/shards"
mkdir -p "$shard_dir"
pids=()
for ((shard=0; shard<EVAL_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" \
        "$ROOT/experiments/robot/calvin/run_calvin_eval.py" \
        --checkpoint "$RUN_DIR" \
        --dataset-root "$DATA_ROOT" \
        --output "$shard_dir/shard-$shard.json" \
        --num-sequences 1000 \
        --shard-index "$shard" \
        --shard-count "$EVAL_SHARDS" \
        --device cuda:0 \
        --exit-threshold 0.55 \
        --max-reasoning-steps 5 \
        >>"$LOG_ROOT/${RUN_ID}.eval.shard${shard}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
"$PYTHON" "$ROOT/scripts/merge_calvin_eval_shards.py" \
    --shard-dir "$shard_dir" \
    --output "$RESULT_ROOT/$RUN_ID/table3_results.json"
