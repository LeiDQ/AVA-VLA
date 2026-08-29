#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
LOG_ROOT="$ROOT/logs/paper_reproduction"
STATUS="$LOG_ROOT/status.log"
mkdir -p "$LOG_ROOT"

exec 7>"$LOG_ROOT/supervisor.lock"
flock -n 7 || exit 0

status() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

status "supervisor_start waiting_for_verified_openvla_base"
while true; do
    base_model=""
    for candidate in \
        "$ROOT/models/openvla-7b-modelscope-prismatic" \
        "$ROOT/models/openvla-7b-prismatic"; do
        if "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$candidate" >/dev/null 2>&1; then
            base_model="$candidate"
            break
        fi
    done
    if [[ -z "$base_model" ]]; then
        sleep 20
        continue
    fi
    status "base_candidate_complete path=$base_model full_hash_verification=start"
    if ! "$PYTHON" "$ROOT/scripts/validate_openvla_base.py" "$base_model" --full-hash \
        >>"$LOG_ROOT/base_validation.log" 2>&1; then
        status "base_candidate_invalid path=$base_model retry_seconds=60"
        sleep 60
        continue
    fi
    status "base_verified path=$base_model"
    break
done

while true; do
    status "source_regression_start"
    if CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
        "$PYTHON" "$ROOT/scripts/test_avavla_safety_regressions.py" \
        >>"$LOG_ROOT/preflight.regressions.log" 2>&1 \
        && CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
        "$PYTHON" "$ROOT/scripts/test_avavla_regressions.py" \
        >>"$LOG_ROOT/preflight.regressions.log" 2>&1 \
        && "$PYTHON" "$ROOT/scripts/test_libero_eval_sharding.py" \
        >>"$LOG_ROOT/preflight.regressions.log" 2>&1; then
        status "source_regression_complete"
    else
        status "source_regression_failed retry_seconds=60"
        sleep 60
        continue
    fi

    status "official_base_preflight_start path=$base_model gpus=8"
    if BASE_MODEL="$base_model" bash "$ROOT/scripts/run_official_base_preflight.sh" "$base_model" \
        >>"$LOG_ROOT/preflight.supervisor.log" 2>&1; then
        status "official_base_preflight_complete path=$base_model"
        break
    fi
    status "official_base_preflight_failed retry_seconds=60 log=$LOG_ROOT/preflight.train.log"
    sleep 60
done

status "formal_pipeline_start base=$base_model suites=4 seeds=3 gpus=8"
export BASE_MODEL="$base_model"
exec bash "$ROOT/scripts/run_paper_libero_per_suite.sh"
