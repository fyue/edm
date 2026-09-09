#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 6-1:
# EDMの前向きノイズ付加 x_sigma = x_0 + sigma eps によるノルム密度図を作る。

cd "$(dirname "$0")/../../../.."

PYTHON="${PYTHON:-python}"
EXP_ROOT="${EXP_ROOT:-experiments/6/1_edm_forward_norm_density}"
SCRIPT="$EXP_ROOT/scripts/plot_edm_forward_norm_density.py"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

DATA="${DATA:-datasets/cifar10-32x32.zip}"
MAX_IMAGES="${MAX_IMAGES:-5000}"
BATCH="${BATCH:-256}"
STEPS="${STEPS:-256}"
SIGMA_MIN="${SIGMA_MIN:-0.002}"
SIGMA_MAX="${SIGMA_MAX:-80}"
RHO="${RHO:-7}"
SIGMA_DATA="${SIGMA_DATA:-0.5}"
SEED="${SEED:-0}"
MODE="${MODE:-both}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
LOG="$LOG_DIR/edm-forward-norm-density.log"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

run_mode() {
    local mode="$1"
    local norm_max_arg=()
    if [[ "$mode" == "scaled" ]]; then
        norm_max_arg=(--norm-max "${NORM_MAX_SCALED:-1.4}")
    elif [[ -n "${NORM_MAX_RAW:-}" ]]; then
        norm_max_arg=(--norm-max "$NORM_MAX_RAW")
    fi

    local out="$RESULTS_DIR/edm_forward_norm_density_${mode}.png"
    log "Creating $out"
    "$PYTHON" "$SCRIPT" \
        --data "$DATA" \
        --out "$out" \
        --mode "$mode" \
        --max-images "$MAX_IMAGES" \
        --batch "$BATCH" \
        --steps "$STEPS" \
        --sigma-min "$SIGMA_MIN" \
        --sigma-max "$SIGMA_MAX" \
        --rho "$RHO" \
        --sigma-data "$SIGMA_DATA" \
        --seed "$SEED" \
        "${norm_max_arg[@]}" 2>&1 | tee -a "$LOG"
}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    log "Missing executable python: $PYTHON"
    exit 1
fi

if [[ ! -f "$DATA" ]]; then
    log "Missing dataset: $DATA"
    exit 1
fi

case "$MODE" in
    both)
        run_mode scaled
        run_mode raw
        ;;
    scaled|raw)
        run_mode "$MODE"
        ;;
    *)
        log "MODE must be one of: both, scaled, raw"
        exit 1
        ;;
esac

log "Done"
