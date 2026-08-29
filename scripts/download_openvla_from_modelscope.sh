#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
HF_DIR="$ROOT/models/openvla-7b-hf-modelscope"
TEMPLATE_DIR="$ROOT/models/openvla-7b-prismatic"
OUTPUT_DIR="$ROOT/models/openvla-7b-modelscope-prismatic"
LOG_DIR="$ROOT/logs/model_download"
STATUS="$LOG_DIR/openvla_modelscope.status"
BASE_URL="https://www.modelscope.cn/models/zixiaoBios/openvla-7b/resolve/master"

mkdir -p "$HF_DIR/.partial" "$OUTPUT_DIR" "$LOG_DIR"
printf 'downloading %s\n' "$(date -Is)" >"$STATUS"

download_metadata() {
    local name="$1"
    local expected_size="$2"
    local expected_sha="$3"
    local destination="$HF_DIR/$name"
    local temporary="$HF_DIR/.partial/$name.metadata-tmp"
    if [[ -f "$destination" ]] \
        && [[ "$(stat -c %s "$destination")" == "$expected_size" ]] \
        && [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected_sha" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "$destination")" "$(dirname "$temporary")"
    curl --fail --location --retry 20 --retry-all-errors --connect-timeout 30 \
        --output "$temporary" "$BASE_URL/$name" || return 1
    [[ "$(stat -c %s "$temporary")" == "$expected_size" ]] || return 1
    [[ "$(sha256sum "$temporary" | awk '{print $1}')" == "$expected_sha" ]] || return 1
    mv "$temporary" "$destination"
}

download_metadata \
    model.safetensors.index.json \
    94764 \
    b3a4516f74a609422002e265e542c23ddf11e54c2b781e1c0d4e23e5181f3fd0 || {
        printf 'failed metadata=index %s\n' "$(date -Is)" >"$STATUS"
        exit 1
    }
download_metadata \
    config.json \
    60712 \
    edd5c5cf6d7927e07465cf086ebe41f7b3ec8f3b128a51f71d6db14dad7ad8b1 || {
        printf 'failed metadata=config %s\n' "$(date -Is)" >"$STATUS"
        exit 1
    }

names=(
    model-00001-of-00003.safetensors
    model-00002-of-00003.safetensors
    model-00003-of-00003.safetensors
)
sizes=(6948961960 6971232040 1162406824)
hashes=(
    10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85
    2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a
    ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef
)

pids=()
for index in "${!names[@]}"; do
    name="${names[$index]}"
    size="${sizes[$index]}"
    hash="${hashes[$index]}"
    output="$HF_DIR/$name"
    partial="$HF_DIR/.partial/$name.incomplete"
    if [[ -f "$output" ]] \
        && [[ "$(stat -c %s "$output")" == "$size" ]] \
        && [[ "$(sha256sum "$output" | awk '{print $1}')" == "$hash" ]]; then
        printf '%s already_complete file=%s\n' "$(date -Is)" "$name"
        continue
    fi
    if [[ -f "$partial" && "$(stat -c %s "$partial")" == "$size" ]]; then
        if [[ "$(sha256sum "$partial" | awk '{print $1}')" == "$hash" ]]; then
            mv "$partial" "$output"
            continue
        fi
        printf 'failed invalid_full_partial file=%s %s\n' "$name" "$(date -Is)" >"$STATUS"
        exit 1
    fi
    if [[ ! -f "$partial" ]]; then
        prefix_tmp="$partial.prefix-tmp"
        curl --fail --location --retry 20 --retry-all-errors --connect-timeout 30 \
            --range 0-0 --output "$prefix_tmp" "$BASE_URL/$name" || exit 1
        [[ "$(stat -c %s "$prefix_tmp")" == "1" ]] || exit 1
        mv "$prefix_tmp" "$partial"
    fi
    "$PYTHON" "$ROOT/scripts/parallel_range_download.py" \
        --url "$BASE_URL/$name" \
        --partial "$partial" \
        --output "$output" \
        --expected-size "$size" \
        --expected-sha256 "$hash" \
        --workers 8 \
        --timeout 60 &
    pids+=("$!")
done

download_rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        download_rc=1
    fi
done
if [[ "$download_rc" -ne 0 ]]; then
    printf 'failed weights %s\n' "$(date -Is)" >"$STATUS"
    exit 1
fi

printf 'converting %s\n' "$(date -Is)" >"$STATUS"
"$PYTHON" "$ROOT/scripts/convert_hf_openvla_to_prismatic.py" \
    --hf-dir "$HF_DIR" \
    --template-dir "$TEMPLATE_DIR" \
    --output-dir "$OUTPUT_DIR" || {
        printf 'failed conversion %s\n' "$(date -Is)" >"$STATUS"
        exit 1
    }
printf 'complete %s\n' "$(date -Is)" >"$STATUS"
