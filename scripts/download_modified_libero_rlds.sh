#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${PROJECT_ROOT}/data/modified_libero_rlds}"
STATE_DIR="${PROJECT_ROOT}/logs/dataset_download"
LOG_FILE="${STATE_DIR}/modified_libero_rlds.log"
STATUS_FILE="${STATE_DIR}/modified_libero_rlds.status"
COMPLETE_FILE="${DATA_DIR}/.download_complete"

mkdir -p "${DATA_DIR}" "${STATE_DIR}"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

attempt=0
while true; do
    attempt=$((attempt + 1))
    started_at="$(date --iso-8601=seconds)"
    printf 'state=running\nattempt=%s\nstarted_at=%s\ntarget=%s\n' \
        "${attempt}" "${started_at}" "${DATA_DIR}" > "${STATUS_FILE}"
    printf '[%s] attempt %s: downloading openvla/modified_libero_rlds to %s\n' \
        "${started_at}" "${attempt}" "${DATA_DIR}" >> "${LOG_FILE}"

    if hf download openvla/modified_libero_rlds \
        --repo-type dataset \
        --local-dir "${DATA_DIR}" >> "${LOG_FILE}" 2>&1; then
        completed_at="$(date --iso-8601=seconds)"
        size="$(du -sh "${DATA_DIR}" | awk '{print $1}')"
        printf 'completed_at=%s\nsize=%s\nsource=https://huggingface.co/datasets/openvla/modified_libero_rlds\n' \
            "${completed_at}" "${size}" > "${COMPLETE_FILE}"
        printf 'state=complete\nattempt=%s\ncompleted_at=%s\ntarget=%s\nsize=%s\n' \
            "${attempt}" "${completed_at}" "${DATA_DIR}" "${size}" > "${STATUS_FILE}"
        printf '[%s] download complete; size=%s\n' "${completed_at}" "${size}" >> "${LOG_FILE}"
        exit 0
    fi

    failed_at="$(date --iso-8601=seconds)"
    printf 'state=retrying\nattempt=%s\nlast_failure=%s\ntarget=%s\nretry_after_seconds=60\n' \
        "${attempt}" "${failed_at}" "${DATA_DIR}" > "${STATUS_FILE}"
    printf '[%s] attempt %s failed; retrying in 60 seconds\n' \
        "${failed_at}" "${attempt}" >> "${LOG_FILE}"
    sleep 60
done
