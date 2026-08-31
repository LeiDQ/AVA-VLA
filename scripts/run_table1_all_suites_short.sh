#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-$ROOT/runs/audit}"
RUN_ID="${RUN_ID:-audit_all_suites_8gpu_200bc_papershape_v9}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/modified_libero_rlds}"
TOKENIZER="${TOKENIZER:-$ROOT/models/llama2-7b-ms-tokenizer}"
BASE_MODEL="${BASE_MODEL:-}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2048}"
PYTHON="$ROOT/.venv/bin/python"

if ! [[ "$SHUFFLE_BUFFER_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHUFFLE_BUFFER_SIZE must be a positive integer" >&2
    exit 2
fi

if [[ -z "$BASE_MODEL" ]]; then
    for candidate in "$ROOT/models/openvla-7b-modelscope-prismatic" "$ROOT/models/openvla-7b-prismatic"; do
        if "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$candidate" >/dev/null 2>&1; then
            BASE_MODEL="$candidate"
            break
        fi
    done
fi
if [[ -z "$BASE_MODEL" ]]; then
    echo "No valid OpenVLA base model found" >&2
    exit 2
fi
for dataset in libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops; do
    test -d "$DATA_ROOT/$dataset/1.0.0"
done

export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

"$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
    "$ROOT/vla-scripts/finetune_avavla.py" \
    --vla_path "$BASE_MODEL" \
    --llm_config_path "$TOKENIZER" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name libero_4_task_suites_no_noops \
    --run_root_dir "$RUN_ROOT" \
    --run_id_override "$RUN_ID" \
    --seed 31 \
    --batch_size 8 \
    --grad_accumulation_steps 1 \
    --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
    --image_aug true \
    --use_wrist_image true \
    --use_proprio true \
    --history_window_size 9 \
    --train_base_vla false \
    --bc_steps 200 \
    --latent_warmup_steps 50 \
    --ppo_effective_batch_size 512 \
    --ppo_rollout_size_per_rank 64 \
    --online_envs_per_rank 8 \
    --online_center_crop true \
    --ppo_environment_steps 4096 \
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
    --action_ppo_coef 0.0 \
    --ppo_target_kl 0.02 \
    --ppo_max_backtracks 12 \
    --max_grad_norm 1.0 \
    --max_reasoning_steps 5 \
    --exit_threshold 0.55 \
    --exit_calibration_steps 20 \
    --exit_calibration_buffer_rollouts 4 \
    --online_num_steps_wait 10 \
    --libero_task_suites libero_spatial,libero_object,libero_goal,libero_10 \
    --require_online_task_rewards true \
    --save_freq 50 \
    --ppo_checkpoint_interval_updates 1 \
    --exit_checkpoint_interval_steps 10 \
    --save_latest_checkpoint_only true \
    --use_wandb false
