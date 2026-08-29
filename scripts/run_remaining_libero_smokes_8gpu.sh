#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${1:-$ROOT/models/openvla-7b-modelscope-prismatic}"
LOG_ROOT="$ROOT/logs/suite_smoke/all_8gpu_100bc"
STATUS_LOG="$LOG_ROOT/status.log"
GPU_LIST="0,1,2,3,4,5,6,7"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/supervisor.pid"

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_LOG"
}

run_suite() {
    local suite="$1"
    local dataset="$2"
    status "suite_start suite=$suite dataset=$dataset gpus=8 profile=100bc"
    bash "$ROOT/scripts/run_libero_suite_smoke.sh" \
        "$suite" "$dataset" "$GPU_LIST" "$BASE_MODEL" 8 100bc
    local rc=$?
    if [[ "$rc" -ne 0 ]]; then
        status "suite_failed suite=$suite exit_code=$rc"
        return "$rc"
    fi
    status "suite_complete suite=$suite gpus=8"
}

status "pipeline_start suites=libero_object,libero_goal,libero_10,libero_spatial gpus=8 mode=sequential profile=100bc"
run_suite libero_object libero_object_no_noops || exit $?
run_suite libero_goal libero_goal_no_noops || exit $?
run_suite libero_10 libero_10_no_noops || exit $?
run_suite libero_spatial libero_spatial_no_noops || exit $?
status "pipeline_complete suites=4 gpus=8 profile=100bc"
