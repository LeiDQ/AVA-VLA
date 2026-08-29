#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model_repo="TRI-ML/prismatic-vlms"
model_file="prism-dinosiglip+7b/checkpoints/latest-checkpoint.pt"
local_dir="$repo_root/models/prismatic_repo"
target="$local_dir/$model_file"
expected_bytes=13620752122
log_dir="$repo_root/logs/model_download"
status_file="$log_dir/prism_dinosiglip.status"

mkdir -p "$local_dir" "$log_dir"

while true; do
    if [[ -f "$target" ]] && [[ "$(stat -c %s "$target")" -eq "$expected_bytes" ]]; then
        printf 'complete\n' > "$status_file"
        printf '%s checkpoint complete: %s bytes\n' "$(date -Is)" "$expected_bytes"
        exit 0
    fi

    printf 'hf_downloading\n' > "$status_file"
    printf '%s starting/resuming checkpoint download\n' "$(date -Is)"
    if HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
       HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}" \
       "$repo_root/.venv/bin/hf" download "$model_repo" "$model_file" --local-dir "$local_dir"; then
        continue
    fi

    printf 'retrying\n' > "$status_file"
    printf '%s download interrupted; retrying in 15 seconds\n' "$(date -Is)" >&2
    sleep 15
done
