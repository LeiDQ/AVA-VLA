#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISION_BACKBONE_PROFILE="${VISION_BACKBONE_PROFILE:-openvla-224}"
case "$VISION_BACKBONE_PROFILE" in
    openvla-224)
        DEFAULT_RUN_ROOT="$ROOT/runs/paper_per_suite"
        DEFAULT_LOG_ROOT="$ROOT/logs/paper_reproduction"
        DEFAULT_RESULT_ROOT="$ROOT/results/paper_per_suite"
        REQUIRE_384PX_BACKBONE=false
        ;;
    dinosiglip-384)
        DEFAULT_RUN_ROOT="$ROOT/runs/paper_per_suite_dinosiglip384"
        DEFAULT_LOG_ROOT="$ROOT/logs/paper_reproduction_dinosiglip384"
        DEFAULT_RESULT_ROOT="$ROOT/results/paper_per_suite_dinosiglip384"
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
STATUS_FILE="$LOG_ROOT/status.log"
BASE_MODEL="${BASE_MODEL:-}"
TOKENIZER="${TOKENIZER:-$ROOT/models/llama2-7b-ms-tokenizer}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/modified_libero_rlds}"
PAPER_SEEDS="${PAPER_SEEDS:-0 1 2}"
PAPER_SUITE_FILTER="${PAPER_SUITE_FILTER:-}"
PYTHON="$ROOT/.venv/bin/python"
EVAL_SHARDS="${EVAL_SHARDS:-8}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-2048}"
BC_SAVE_FREQ="${BC_SAVE_FREQ:-1000}"

for integer_setting in EVAL_SHARDS SHUFFLE_BUFFER_SIZE BC_SAVE_FREQ; do
    integer_value="${!integer_setting}"
    if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer (received %s)\n' \
            "$integer_setting" "$integer_value" >&2
        exit 2
    fi
done

if [[ "$VISION_BACKBONE_PROFILE" == "dinosiglip-384" && -z "$BASE_MODEL" ]]; then
    printf '%s\n' \
        'VISION_BACKBONE_PROFILE=dinosiglip-384 requires BASE_MODEL to point to a verified 384px OXE robot base.' >&2
    exit 2
fi
if [[ "$VISION_BACKBONE_PROFILE" == "openvla-224" && -z "$BASE_MODEL" ]]; then
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
BASE_CHECKPOINT="$BASE_MODEL/checkpoints/step-295000-epoch-40-loss=0.2200.pt"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
exec 9>"$LOG_ROOT/pipeline.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" >"$LOG_ROOT/pipeline.pid"

export LIBERO_CONFIG_PATH="$ROOT/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$ROOT/third_party/LIBERO:$ROOT"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_FILE"
}

is_safe_run() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = root / "avavla_config.json"
manifest_path = root / "CHECKPOINT_COMPLETE.json"
if not config_path.is_file() or not manifest_path.is_file():
    raise SystemExit(1)
try:
    with config_path.open() as stream:
        config = json.load(stream)
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if int(config.get("implementation_version", 0)) < 9:
        raise SystemExit(1)
    if int(manifest.get("implementation_version", 0)) < 9:
        raise SystemExit(1)
    required = manifest.get("required_files", {})
    if not isinstance(required, dict) or not required:
        raise SystemExit(1)
    for relative_name, expected_size in required.items():
        artifact = root / relative_name
        if not artifact.is_file() or artifact.stat().st_size != int(expected_size):
            raise SystemExit(1)
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

validate_evaluation() {
    "$PYTHON" - "$1" "$2" "$3" "$4" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
suite = sys.argv[2]
checkpoint = sys.argv[3]
shard_count = int(sys.argv[4])
with path.open() as stream:
    result = json.load(stream)
episodes = int(result["total_episodes"])
successes = int(result["total_successes"])
assert result["task_suite"] == suite
assert result["checkpoint"] == checkpoint
assert int(result["seed"]) == 7
assert int(result["trial_shard_count"]) == shard_count
assert episodes == 500
assert 0 <= successes <= episodes
assert math.isclose(float(result["success_rate"]), successes / episodes, rel_tol=0.0, abs_tol=1e-12)
PY
}
validate_evaluation_shard() {
    "$PYTHON" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
suite = sys.argv[2]
shard_index = int(sys.argv[3])
shard_count = int(sys.argv[4])
checkpoint = sys.argv[5]
with path.open() as stream:
    result = json.load(stream)
expected_indices = list(range(shard_index, 50, shard_count))
expected_episodes = 10 * len(expected_indices)
episodes = int(result["total_episodes"])
successes = int(result["total_successes"])
assert result["task_suite"] == suite
assert result["checkpoint"] == checkpoint
assert int(result["seed"]) == 7
assert int(result["num_tasks"]) == 10
assert int(result["num_trials_per_task"]) == 50
assert int(result["trial_shard_index"]) == shard_index
assert int(result["trial_shard_count"]) == shard_count
assert [int(value) for value in result["trial_indices"]] == expected_indices
assert episodes == expected_episodes
assert 0 <= successes <= episodes
assert math.isclose(float(result["success_rate"]), successes / episodes, rel_tol=0.0, abs_tol=1e-12)
PY
}


if [[ "$VISION_BACKBONE_PROFILE" == "dinosiglip-384" ]]; then
    base_validation=(
        "$PYTHON" "$ROOT/scripts/validate_avavla_vision_base.py"
        "$BASE_MODEL" --expected-resolution 384 --full-hash
    )
else
    base_validation=("$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$BASE_MODEL" --full-hash)
fi
if ! "${base_validation[@]}" >/dev/null 2>&1; then
    status "blocked_invalid_base path=$BASE_MODEL vision_profile=$VISION_BACKBONE_PROFILE"
    exit 2
fi

if (( EVAL_SHARDS > 8 )); then
    status "blocked_invalid_eval_shards value=$EVAL_SHARDS expected=1..8"
    exit 2
fi

datasets=(
    libero_spatial_no_noops
    libero_object_no_noops
    libero_goal_no_noops
    libero_10_no_noops
)
suites=(libero_spatial libero_object libero_goal libero_10)

status "pipeline_start paper=2606.15099 policies=per_suite seeds=[$PAPER_SEEDS] vision_profile=${VISION_BACKBONE_PROFILE:-openvla-224} shuffle_buffer_per_rank=$SHUFFLE_BUFFER_SIZE bc_save_freq=$BC_SAVE_FREQ"
for seed in $PAPER_SEEDS; do
    for index in "${!datasets[@]}"; do
        dataset="${datasets[$index]}"
        suite="${suites[$index]}"
        if [[ -n "$PAPER_SUITE_FILTER" && " $PAPER_SUITE_FILTER " != *" $suite "* ]]; then
            continue
        fi
        short_name="${suite#libero_}"
        run_id="paper_${short_name}_seed${seed}"
        run_dir="$RUN_ROOT/$run_id"
        train_log="$LOG_ROOT/${suite}.seed${seed}.train.log"
        eval_dir="$RESULT_ROOT/$suite/seed${seed}"
        eval_stdout="$LOG_ROOT/${suite}.seed${seed}.eval.log"

        while [[ ! -d "$DATA_ROOT/$dataset/1.0.0" ]]; do
            status "waiting_for_dataset suite=$suite dataset=$dataset"
            sleep 300
        done

        if [[ -d "$run_dir" && ! -f "$run_dir/CHECKPOINT_COMPLETE.json" && ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
            archive="$RUN_ROOT/invalid_uncheckpointed/${run_id}.$(date -u +%Y%m%dT%H%M%SZ)"
            mkdir -p "$(dirname "$archive")"
            mv "$run_dir" "$archive"
            status "archived_uncheckpointed_run run=$run_id archive=$archive reason=no_atomic_manifest"
        elif [[ -d "$run_dir" ]] && ! is_safe_run "$run_dir"; then
            archive="$RUN_ROOT/invalid_incompatible/${run_id}.$(date -u +%Y%m%dT%H%M%SZ)"
            mkdir -p "$(dirname "$archive")"
            mv "$run_dir" "$archive"
            status "archived_invalid_checkpoint run=$run_id archive=$archive reason=incompatible_checkpoint_contract"
        fi

        while true; do
            if [[ -f "$run_dir/TRAINING_COMPLETE" ]] && is_safe_run "$run_dir"; then
                break
            fi
            if [[ -d "$run_dir" ]] && ! is_safe_run "$run_dir"; then
                archive="$RUN_ROOT/invalid_retry/${run_id}.$(date -u +%Y%m%dT%H%M%S%N)"
                mkdir -p "$(dirname "$archive")"
                mv "$run_dir" "$archive"
                status "archived_invalid_retry run=$run_id archive=$archive reason=missing_or_incompatible_manifest"
            fi

            resume_args=(--vla_path "$BASE_MODEL")
            if [[ -f "$run_dir/CHECKPOINT_COMPLETE.json" ]] && is_safe_run "$run_dir"; then
                resume_args=(--vla_path "$run_dir" --resume true)
            fi
            status "training_start suite=$suite dataset=$dataset seed=$seed resume=${resume_args[*]}"
            "$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
                "$ROOT/vla-scripts/finetune_avavla.py" \
                "${resume_args[@]}" \
                --llm_config_path "$TOKENIZER" \
                --data_root_dir "$DATA_ROOT" \
                --dataset_name "$dataset" \
                --run_root_dir "$RUN_ROOT" \
                --run_id_override "$run_id" \
                --seed "$seed" \
                --batch_size 8 \
                --grad_accumulation_steps 1 \
                --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE" \
                --image_aug true \
                --use_wrist_image true \
                --use_proprio true \
                --require_384px_backbone "${REQUIRE_384PX_BACKBONE:-false}" \
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
                --libero_task_suites "$suite" \
                --require_online_task_rewards true \
                --save_freq "$BC_SAVE_FREQ" \
                --ppo_checkpoint_interval_updates 10 \
                --exit_checkpoint_interval_steps 1000 \
                --save_latest_checkpoint_only true \
                --use_wandb false >>"$train_log" 2>&1
            rc=$?
            if [[ -f "$run_dir/TRAINING_COMPLETE" ]] && is_safe_run "$run_dir"; then
                status "training_complete suite=$suite seed=$seed run=$run_id"
                break
            fi
            if [[ -d "$run_dir" ]] && ! is_safe_run "$run_dir"; then
                status "training_invalid_checkpoint suite=$suite seed=$seed exit_code=$rc retry=immediate"
                continue
            fi
            status "training_retry suite=$suite seed=$seed exit_code=$rc delay_seconds=60"
            sleep 60
        done

        if ! "$PYTHON" "$ROOT/scripts/validate_avavla_stage_checkpoints.py" \
            "$run_dir" --write-report >>"$train_log" 2>&1; then
            status "stage_checkpoint_validation_failed suite=$suite seed=$seed run=$run_id"
            exit 5
        fi
        status "stage_checkpoints_validated suite=$suite seed=$seed run=$run_id"

        curve_dir="$RESULT_ROOT/$suite/seed${seed}/training_curves"
        if "$PYTHON" "$ROOT/scripts/plot_avavla_losses.py" \
            "$run_dir/metrics.jsonl" --output-dir "$curve_dir" \
            >>"$LOG_ROOT/loss_curves.log" 2>&1; then
            status "loss_curves_complete suite=$suite seed=$seed path=$curve_dir/loss_curves.png"
        else
            status "loss_curves_failed suite=$suite seed=$seed log=$LOG_ROOT/loss_curves.log"
        fi

        mkdir -p "$eval_dir"
        while ! validate_evaluation "$eval_dir/evaluation_results.json" "$suite" "$run_dir" "$EVAL_SHARDS" 2>/dev/null; do
            rm -f "$eval_dir/EVALUATION_COMPLETE"
            status "evaluation_start suite=$suite seed=$seed trials=500 shards=$EVAL_SHARDS checkpoint=$run_dir"
            eval_pids=()
            eval_pid_shards=()
            shard_results=()
            for ((shard=0; shard<EVAL_SHARDS; shard++)); do
                shard_dir="$eval_dir/shard_$shard"
                shard_result="$shard_dir/evaluation_results.json"
                shard_results+=("$shard_result")
                if validate_evaluation_shard "$shard_result" "$suite" "$shard" "$EVAL_SHARDS" "$run_dir" 2>/dev/null; then
                    status "evaluation_shard_reused suite=$suite seed=$seed shard=$shard"
                    continue
                fi
                mkdir -p "$shard_dir"
                shard_log="$LOG_ROOT/${suite}.seed${seed}.eval.shard${shard}.log"
                status "evaluation_shard_start suite=$suite seed=$seed shard=$shard gpu=$shard"
                CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" \
                    "$ROOT/experiments/robot/libero/run_libero_eval.py" \
                    --model_family avavla \
                    --pretrained_checkpoint "$run_dir" \
                    --use_l1_regression true \
                    --num_images_in_input 2 \
                    --use_proprio true \
                    --enable_latent_reasoning true \
                    --max_reasoning_steps 5 \
                    --exit_threshold 0.55 \
                    --use_history_state true \
                    --center_crop true \
                    --num_open_loop_steps 8 \
                    --task_suite_name "$suite" \
                    --num_steps_wait 10 \
                    --num_trials_per_task 50 \
                    --trial_shard_index "$shard" \
                    --trial_shard_count "$EVAL_SHARDS" \
                    --initial_states_path DEFAULT \
                    --env_img_res 256 \
                    --local_log_dir "$shard_dir" \
                    --save_video false \
                    --use_wandb false \
                    --seed 7 >>"$shard_log" 2>&1 &
                eval_pids+=("$!")
                eval_pid_shards+=("$shard")
            done
            rc=0
            for pid_index in "${!eval_pids[@]}"; do
                if ! wait "${eval_pids[$pid_index]}"; then
                    rc=1
                    status "evaluation_shard_failed suite=$suite seed=$seed shard=${eval_pid_shards[$pid_index]}"
                fi
            done
            if [[ $rc -eq 0 ]]; then
                "$PYTHON" "$ROOT/scripts/merge_libero_eval_shards.py" \
                    --output "$eval_dir/evaluation_results.json" \
                    "${shard_results[@]}" >>"$eval_stdout" 2>&1
                rc=$?
            fi
            if [[ $rc -eq 0 ]] && validate_evaluation "$eval_dir/evaluation_results.json" "$suite" "$run_dir" "$EVAL_SHARDS"; then
                result_line="$("$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("{:.2f}% ({}/{})".format(100*r["success_rate"], r["total_successes"], r["total_episodes"]))' "$eval_dir/evaluation_results.json")"
                status "evaluation_complete suite=$suite seed=$seed result=$result_line"
                break
            fi
            status "evaluation_retry suite=$suite seed=$seed exit_code=$rc delay_seconds=60"
            sleep 60
        done
    done
done

if [[ -z "$PAPER_SUITE_FILTER" ]]; then
    "$PYTHON" "$ROOT/scripts/aggregate_table1_ours.py" \
        --result-root "$RESULT_ROOT" \
        --mode per_suite \
        --seeds $PAPER_SEEDS \
        --output "$RESULT_ROOT/table1_ours_per_suite.json" \
        >>"$LOG_ROOT/aggregate_table1.log" 2>&1
    status "table1_aggregate_complete mode=per_suite path=$RESULT_ROOT/table1_ours_per_suite.json"
fi

status "pipeline_complete suite_filter=[$PAPER_SUITE_FILTER] seeds=[$PAPER_SEEDS]"
date -Is >"$RESULT_ROOT/ALL_COMPLETE"
