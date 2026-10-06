#!/usr/bin/env bash
set -Eeuo pipefail

# Wait until a physical GPU has no compute processes, then run one experiment.
# Configure the experiment through RUN_SCRIPT and optionally EXP_ROOT.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_SCRIPT="${RUN_SCRIPT:?Set RUN_SCRIPT to the experiment run script}"
EXP_ROOT="${EXP_ROOT:-$(dirname "$(dirname "$RUN_SCRIPT")")}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
WAIT_LOG="${WAIT_LOG:-$LOG_DIR/schedule-after-gpu-free.log}"
GPU_INDEX="${GPU_INDEX:-0}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-Quadro RTX 8000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-$(basename "$EXP_ROOT")}"
PLAN_ONLY="${PLAN_ONLY:-0}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$WAIT_LOG"
}

gpu_compute_processes() {
    nvidia-smi -i "$GPU_INDEX" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d' || true
}

verify_gpu_identity() {
    local actual_name
    actual_name="$(nvidia-smi -i "$GPU_INDEX" \
        --query-gpu=name --format=csv,noheader,nounits 2>/dev/null \
        | sed -n '1p')"
    if [[ -z "$actual_name" ]]; then
        log "Could not query GPU index: $GPU_INDEX"
        log "Available GPUs:"
        nvidia-smi --query-gpu=index,name \
            --format=csv,noheader,nounits 2>/dev/null | tee -a "$WAIT_LOG"
        return 1
    fi
    if [[ -n "$EXPECTED_GPU_NAME" && "$actual_name" != "$EXPECTED_GPU_NAME" ]]; then
        log "GPU index ${GPU_INDEX} is '${actual_name}', expected '${EXPECTED_GPU_NAME}'"
        log "Check the physical GPU 0 identity; repository policy prohibits switching to another GPU."
        return 1
    fi
    log "Target GPU: index ${GPU_INDEX} (${actual_name})"
}

main() {
    local active

    if [[ ! -x "$RUN_SCRIPT" ]]; then
        printf 'Missing executable run script: %s\n' "$RUN_SCRIPT" >&2
        return 1
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf 'PLAN: skip GPU wait and validate %s\n' "$RUN_SCRIPT"
        exec env PLAN_ONLY=1 "$RUN_SCRIPT" "$@"
    fi
    if [[ "$GPU_INDEX" != 0 ]]; then
        printf 'GPU policy: only physical GPU 0 is allowed (GPU_INDEX=%s).\n' "$GPU_INDEX" >&2
        return 1
    fi
    mkdir -p "$LOG_DIR"
    if [[ ! "$CHECK_INTERVAL_SEC" =~ ^[1-9][0-9]*$ ]]; then
        log "CHECK_INTERVAL_SEC must be a positive integer: $CHECK_INTERVAL_SEC"
        return 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "Missing nvidia-smi; cannot wait for GPU ${GPU_INDEX}"
        return 1
    fi

    verify_gpu_identity
    log "Scheduling ${SCHEDULE_LABEL} after GPU ${GPU_INDEX} becomes free"
    log "Run script: $RUN_SCRIPT"
    log "Check interval: ${CHECK_INTERVAL_SEC}s"

    while true; do
        active="$(gpu_compute_processes)"
        if [[ -z "$active" ]]; then
            break
        fi
        log "GPU ${GPU_INDEX} is still busy:"
        printf '%s\n' "$active" | tee -a "$WAIT_LOG"
        sleep "$CHECK_INTERVAL_SEC"
    done

    log "GPU ${GPU_INDEX} is free; launching ${SCHEDULE_LABEL}"
    export GPU_INDEX=0
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    exec "$RUN_SCRIPT" "$@"
}

main "$@"
