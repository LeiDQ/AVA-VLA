#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/models/openvla-7b-prismatic"
LOG_DIR="$ROOT/logs/model_download"
STATUS_FILE="$LOG_DIR/openvla_7b_prismatic.status"
REPO_ID="openvla/openvla-7b-prismatic"
CHECKPOINT="$TARGET/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
EXPECTED_SIZE=30165309772
EXPECTED_SHA256="2c2497cd9e0ecced65e54b0771172f22e3ed64d0c0af339e094349715d3b3602"
INCOMPLETE="$TARGET/.cache/huggingface/download/checkpoints/pTNpmAmpflHovd3Fx6Opt_w6Ph0=.$EXPECTED_SHA256.incomplete"
DOWNLOAD_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CHECKPOINT_URL="${DOWNLOAD_ENDPOINT%/}/$REPO_ID/resolve/main/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
PARALLEL_WORKERS="${PARALLEL_DOWNLOAD_WORKERS:-24}"

mkdir -p "$TARGET" "$LOG_DIR"
while true; do
    if [[ -s "$TARGET/config.json" && -s "$TARGET/dataset_statistics.json" && -s "$CHECKPOINT" ]]; then
        printf 'verifying\n' >"$STATUS_FILE"
        if "$ROOT/.venv/bin/python" "$ROOT/scripts/validate_openvla_base.py" \
            "$TARGET" --full-hash --write-native-manifest; then
            printf 'complete\n' >"$STATUS_FILE"
            printf '%s OpenVLA Prismatic checkpoint complete and verified\n' "$(date -Is)"
            exit 0
        fi
        printf 'failed_integrity\n' >"$STATUS_FILE"
        printf '%s OpenVLA Prismatic checkpoint failed integrity validation\n' "$(date -Is)" >&2
        exit 1
    fi

    printf 'downloading\n' >"$STATUS_FILE"
    if [[ -s "$INCOMPLETE" ]]; then
        printf '%s parallel resume from %s bytes with %s workers\n' \
            "$(date -Is)" "$(stat -c %s "$INCOMPLETE")" "$PARALLEL_WORKERS"
        if "$ROOT/.venv/bin/python" "$ROOT/scripts/parallel_range_download.py" \
            --url "$CHECKPOINT_URL" \
            --partial "$INCOMPLETE" \
            --output "$CHECKPOINT" \
            --expected-size "$EXPECTED_SIZE" \
            --expected-sha256 "$EXPECTED_SHA256" \
            --workers "$PARALLEL_WORKERS" \
            --timeout "${HF_HUB_DOWNLOAD_TIMEOUT:-60}"; then
            continue
        fi
        printf 'retrying\n' >"$STATUS_FILE"
        printf '%s parallel download interrupted; retrying in 15 seconds\n' "$(date -Is)" >&2
        sleep 15
        continue
    fi

    if HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
       HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
       HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}" \
       "$ROOT/.venv/bin/hf" download "$REPO_ID" \
           config.json dataset_statistics.json \
           checkpoints/step-295000-epoch-40-loss=0.2200.pt \
           --local-dir "$TARGET"; then
        continue
    fi

    printf 'retrying\n' >"$STATUS_FILE"
    printf '%s download interrupted; retrying in 15 seconds\n' "$(date -Is)" >&2
    sleep 15
done
