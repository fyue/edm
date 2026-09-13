#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/edm-exp7-5-matplotlib}"

PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
EXP_ROOT="${EXP_ROOT:-experiments/7/5_fm_edm_trajectory_error}"
CONFIG="${CONFIG:-$EXP_ROOT/config/pilot.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
PLAN_ONLY="${PLAN_ONLY:-0}"
LABEL_FILTER="${LABEL_FILTER:-}"
DEVICE="${DEVICE:-cuda}"
SECONDS_PER_NFE_IMAGE="${SECONDS_PER_NFE_IMAGE:-0.00244}"

MASTER_LOG="$LOG_DIR/trajectory-error.log"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR" "$MPLCONFIGDIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

check_inputs() {
    local missing=0
    if [[ ! -x "$PYTHON" ]]; then
        log "Missing executable python: $PYTHON"
        missing=1
    fi
    if [[ ! -f "$CONFIG" ]]; then
        log "Missing config: $CONFIG"
        missing=1
    fi
    if (( missing != 0 )); then
        exit 1
    fi
}

run_stage() {
    local label="$1"
    local fm_network="$2"
    local edm_network="$3"
    local seeds="$4"
    local batch="$5"
    local ref_coarse_steps="$6"
    local ref_fine_steps="$7"
    local outdir="$RESULTS_ROOT/$label"
    local run_log="$LOG_DIR/$label.log"
    local -a command=(
        "$PYTHON"
        "$EXP_ROOT/scripts/trajectory_error.py"
        --fm-network "$fm_network"
        --edm-network "$edm_network"
        --seeds "$seeds"
        --batch "$batch"
        --ref-coarse-steps "$ref_coarse_steps"
        --ref-fine-steps "$ref_fine_steps"
        --outdir "$outdir"
        --device "$DEVICE"
        --seconds-per-nfe-image "$SECONDS_PER_NFE_IMAGE"
    )

    if [[ -n "$LABEL_FILTER" && "$label" != "$LABEL_FILTER" ]]; then
        log "[$label] filtered out by LABEL_FILTER; skipping"
        return
    fi
    if [[ "$PLAN_ONLY" != "1" && -f "$outdir/_SUCCESS" ]]; then
        log "[$label] complete marker exists; skipping"
        return
    fi
    if [[ ! -f "$fm_network" ]]; then
        log "[$label] missing FM checkpoint: $fm_network"
        return 1
    fi
    if [[ ! -f "$edm_network" ]]; then
        log "[$label] missing EDM checkpoint: $edm_network"
        return 1
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        command+=(--plan-only)
        log "PLAN [$label seeds=$seeds batch=$batch RK4=$ref_coarse_steps/$ref_fine_steps]"
    else
        log "[$label] starting FM/EDM trajectory analysis"
    fi
    "${command[@]}" 2>&1 | tee -a "$run_log" "$MASTER_LOG"
    if [[ "$PLAN_ONLY" != "1" ]]; then
        log "[$label] completed; results: $outdir"
    fi
}

main() {
    check_inputs
    log "Experiment 7-5: FM/EDM trajectory and integration error"
    log "Config: $CONFIG"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no GPU work will be started"
    fi

    local label
    local fm_network
    local edm_network
    local seeds
    local batch
    local ref_coarse_steps
    local ref_fine_steps
    local selected=0

    while IFS=$'\t' read -r label fm_network edm_network seeds batch ref_coarse_steps ref_fine_steps; do
        [[ -z "${label:-}" || "$label" =~ ^# ]] && continue
        selected=$((selected + 1))
        run_stage "$label" "$fm_network" "$edm_network" "$seeds" "$batch" \
            "$ref_coarse_steps" "$ref_fine_steps"
        if [[ -f "$STOP_FILE" ]]; then
            log "Stop file found: $STOP_FILE"
            return
        fi
    done < "$CONFIG"

    if (( selected == 0 )); then
        log "No stages found in $CONFIG"
        exit 1
    fi
}

main "$@"
