#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/edm-matplotlib}"

PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
EXP_ROOT="${EXP_ROOT:-experiments/5/13_trajectory_distance_5-4_vs_5-11}"
CONFIG="${CONFIG:-$EXP_ROOT/config/pilot.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
PLAN_ONLY="${PLAN_ONLY:-0}"
LABEL_FILTER="${LABEL_FILTER:-}"
DEVICE="${DEVICE:-cuda}"
SECONDS_PER_NFE_IMAGE="${SECONDS_PER_NFE_IMAGE:-0.00244}"

MASTER_LOG="$LOG_DIR/trajectory-distance.log"
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

run_condition() {
    local label="$1"
    local network="$2"
    local seeds="$3"
    local batch="$4"
    local steps="$5"
    local sigma_min="$6"
    local sigma_max="$7"
    local rho="$8"
    local rk_subdivisions="$9"
    local outdir="$RESULTS_ROOT/$label"
    local run_log="$LOG_DIR/$label.log"
    local -a command=(
        "$PYTHON"
        "$EXP_ROOT/scripts/trajectory_distance.py"
        --network "$network"
        --seeds "$seeds"
        --batch "$batch"
        --steps "$steps"
        --sigma-min "$sigma_min"
        --sigma-max "$sigma_max"
        --rho "$rho"
        --rk-subdivisions "$rk_subdivisions"
        --outdir "$outdir"
        --device "$DEVICE"
        --seconds-per-nfe-image "$SECONDS_PER_NFE_IMAGE"
    )

    if [[ -n "$LABEL_FILTER" && "$label" != "$LABEL_FILTER" ]]; then
        log "[$label] Filtered out by LABEL_FILTER; skipping"
        return
    fi
    if [[ "$PLAN_ONLY" != "1" && -f "$outdir/_SUCCESS" ]]; then
        log "[$label] Complete marker exists; skipping"
        return
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        command+=(--plan-only)
        log "PLAN [$label seeds=$seeds steps=$steps RK4=$rk_subdivisions]"
    else
        log "[$label] Starting trajectory-distance measurement"
    fi
    "${command[@]}" 2>&1 | tee -a "$run_log" "$MASTER_LOG"
    if [[ "$PLAN_ONLY" != "1" ]]; then
        log "[$label] Completed; results: $outdir"
    fi
}

main() {
    check_inputs
    log "Config: $CONFIG"
    log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    local label
    local network
    local seeds
    local batch
    local steps
    local sigma_min
    local sigma_max
    local rho
    local rk_subdivisions
    while IFS=$'\t' read -r label network seeds batch steps sigma_min sigma_max rho rk_subdivisions; do
        [[ -z "${label:-}" || "$label" =~ ^# ]] && continue
        run_condition \
            "$label" "$network" "$seeds" "$batch" "$steps" \
            "$sigma_min" "$sigma_max" "$rho" "$rk_subdivisions"
        if [[ -f "$STOP_FILE" ]]; then
            log "Stop file found: $STOP_FILE"
            return
        fi
    done < "$CONFIG"
}

main "$@"
