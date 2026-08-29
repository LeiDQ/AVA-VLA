#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${1:-$ROOT/models/openvla-7b-modelscope-prismatic}"
WAIT_STATUS="$ROOT/logs/suite_smoke/remaining_8gpu_5min/status.log"
LOG_ROOT="$ROOT/logs/suite_smoke/object_curve_8gpu_queue"
STATUS_LOG="$LOG_ROOT/status.log"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/supervisor.pid"

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS_LOG"
}

status "waiting_for_remaining_smokes"
while true; do
    line=$(tail -n 1 "$WAIT_STATUS" 2>/dev/null || true)
    case "$line" in
        *pipeline_complete*) break ;;
        *suite_failed*)
            status "blocked reason=remaining_smoke_pipeline_failed"
            exit 1
            ;;
    esac
    sleep 15
done

status "start suite=libero_object profile=object_curve gpus=8"
bash "$ROOT/scripts/run_libero_suite_smoke.sh" \
    libero_object libero_object_no_noops 0,1,2,3,4,5,6,7 "$BASE_MODEL" 8 object_curve
rc=$?
if [[ "$rc" -ne 0 ]]; then
    status "failed suite=libero_object exit_code=$rc"
    exit "$rc"
fi
status "complete suite=libero_object profile=object_curve gpus=8"
