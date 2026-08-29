#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/paper_reproduction}"
PID_FILE="${PID_FILE:-$LOG_ROOT/pipeline.pid}"
GPU_LOG="$LOG_ROOT/gpu_telemetry.csv"
PROCESS_LOG="$LOG_ROOT/process_telemetry.csv"
HOST_MEMORY_LOG="$LOG_ROOT/host_memory_telemetry.csv"
mkdir -p "$LOG_ROOT"

exec 8>"$LOG_ROOT/telemetry.lock"
if ! flock -n 8; then
    exit 0
fi

if [[ ! -s "$GPU_LOG" ]]; then
    printf '%s\n' 'timestamp,gpu_index,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c' >"$GPU_LOG"
fi
if [[ ! -s "$PROCESS_LOG" ]]; then
    printf '%s\n' 'timestamp,pipeline_pid,torchrun_processes,training_workers,train_log_bytes,filesystem_free_bytes' >"$PROCESS_LOG"
fi
if [[ ! -s "$HOST_MEMORY_LOG" ]]; then
    printf '%s\n' 'timestamp,mem_available_bytes,cgroup_usage_bytes,cgroup_limit_bytes,training_workers_rss_bytes' >"$HOST_MEMORY_LOG"
fi

while true; do
    pipeline_pid=""
    if [[ -s "$PID_FILE" ]]; then
        read -r pipeline_pid <"$PID_FILE" || true
    fi
    if [[ -z "$pipeline_pid" ]] || ! kill -0 "$pipeline_pid" 2>/dev/null; then
        sleep 10
        if [[ -z "$pipeline_pid" ]] || ! kill -0 "$pipeline_pid" 2>/dev/null; then
            exit 0
        fi
    fi

    timestamp="$(date -Is)"
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits \
        | while IFS= read -r row; do
            printf '%s,%s\n' "$timestamp" "${row// /}"
        done >>"$GPU_LOG" 2>>"$LOG_ROOT/telemetry.err.log"

    torchrun_processes="$(pgrep -fc 'torchrun.*finetune_avavla.py' || true)"
    training_workers="$(pgrep -fc '[f]inetune_avavla.py' || true)"
    train_log_bytes="$(find "$LOG_ROOT" -maxdepth 1 -type f -name 'libero_*.train.log' -printf '%s\n' 2>/dev/null | awk '{total += $1} END {print total + 0}')"
    filesystem_free_bytes="$(df -PB1 "$ROOT" | awk 'NR == 2 {print $4}')"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$timestamp" "$pipeline_pid" "$torchrun_processes" "$training_workers" \
        "$train_log_bytes" "$filesystem_free_bytes" >>"$PROCESS_LOG"

    mem_available_bytes="$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)"
    if [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
        read -r cgroup_usage_bytes </sys/fs/cgroup/memory/memory.usage_in_bytes
        read -r cgroup_limit_bytes </sys/fs/cgroup/memory/memory.limit_in_bytes
    else
        read -r cgroup_usage_bytes </sys/fs/cgroup/memory.current
        read -r cgroup_limit_bytes </sys/fs/cgroup/memory.max
    fi
    worker_rss_kib="$({
        for worker_pid in $(pgrep -f "$ROOT/vla-scripts/finetune_avavla.py" || true); do
            awk '/^VmRSS:/ {print $2}' "/proc/$worker_pid/status" 2>/dev/null || true
        done
    } | awk '{total += $1} END {printf "%.0f", total + 0}')"
    printf '%s,%s,%s,%s,%s\n' \
        "$timestamp" "$mem_available_bytes" "$cgroup_usage_bytes" "$cgroup_limit_bytes" \
        "$((worker_rss_kib * 1024))" >>"$HOST_MEMORY_LOG"
    sleep 30
done
