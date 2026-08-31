#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-$ROOT/runs/paper_per_suite_corrected/paper_spatial_seed0/stage_checkpoints/latent_warmup_complete}"
OUTPUT_ROOT="${2:-$ROOT/results/warmup_validation/spatial_seed7_trials4_fixed_steps}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-4}"
SHARD_COUNT="${SHARD_COUNT:-4}"

if [[ ! -f "$CHECKPOINT/CHECKPOINT_COMPLETE.json" ]]; then
    printf 'Missing complete checkpoint manifest: %s\n' "$CHECKPOINT" >&2
    exit 2
fi
if [[ "$SHARD_COUNT" -ne 4 ]]; then
    printf 'This paired 8-GPU canary requires SHARD_COUNT=4 (received %s)\n' "$SHARD_COUNT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export LIBERO_CONFIG_PATH="$ROOT/.libero"
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export OMP_NUM_THREADS=4

pids=()
labels=()
for ((shard=0; shard<SHARD_COUNT; shard++)); do
    zero_step_dir="$OUTPUT_ROOT/reasoning_0step/shard_$shard"
    reasoning_dir="$OUTPUT_ROOT/reasoning_5step/shard_$shard"
    mkdir -p "$zero_step_dir" "$reasoning_dir"

    if [[ "$shard" -eq 0 ]]; then
        visible_devices=0
    else
        # PyTorch uses the first visible GPU as cuda:0.  Keeping physical GPU
        # 0 second satisfies robosuite's legacy EGL environment assertion,
        # while EGL device 0 still maps to the process-local primary GPU.
        visible_devices="$shard,0"
    fi
    CUDA_VISIBLE_DEVICES="$visible_devices" MUJOCO_EGL_DEVICE_ID=0 "$ROOT/.venv/bin/python" \
        "$ROOT/experiments/robot/libero/run_libero_eval.py" \
        --model_family avavla \
        --pretrained_checkpoint "$CHECKPOINT" \
        --use_l1_regression true \
        --num_images_in_input 2 \
        --use_proprio true \
        --enable_latent_reasoning true \
        --max_reasoning_steps 5 \
        --fixed_reasoning_steps 0 \
        --use_history_state true \
        --center_crop true \
        --num_open_loop_steps 8 \
        --task_suite_name libero_spatial \
        --num_steps_wait 10 \
        --num_trials_per_task "$TRIALS_PER_TASK" \
        --trial_shard_index "$shard" \
        --trial_shard_count "$SHARD_COUNT" \
        --initial_states_path DEFAULT \
        --env_img_res 256 \
        --local_log_dir "$zero_step_dir" \
        --save_video false \
        --use_wandb false \
        --seed 7 >"$zero_step_dir/stdout.log" 2>&1 &
    pids+=("$!")
    labels+=("reasoning_0step:$shard")

    gpu=$((shard + SHARD_COUNT))
    visible_devices="$gpu,0"
    CUDA_VISIBLE_DEVICES="$visible_devices" MUJOCO_EGL_DEVICE_ID=0 "$ROOT/.venv/bin/python" \
        "$ROOT/experiments/robot/libero/run_libero_eval.py" \
        --model_family avavla \
        --pretrained_checkpoint "$CHECKPOINT" \
        --use_l1_regression true \
        --num_images_in_input 2 \
        --use_proprio true \
        --enable_latent_reasoning true \
        --max_reasoning_steps 5 \
        --fixed_reasoning_steps 5 \
        --use_history_state true \
        --center_crop true \
        --num_open_loop_steps 8 \
        --task_suite_name libero_spatial \
        --num_steps_wait 10 \
        --num_trials_per_task "$TRIALS_PER_TASK" \
        --trial_shard_index "$shard" \
        --trial_shard_count "$SHARD_COUNT" \
        --initial_states_path DEFAULT \
        --env_img_res 256 \
        --local_log_dir "$reasoning_dir" \
        --save_video false \
        --use_wandb false \
        --seed 7 >"$reasoning_dir/stdout.log" 2>&1 &
    pids+=("$!")
    labels+=("reasoning_5step:$shard")
done

rc=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '%s rc=0\n' "${labels[$index]}"
    else
        one_rc=$?
        printf '%s rc=%s\n' "${labels[$index]}" "$one_rc"
        rc=1
    fi
done
exit "$rc"
