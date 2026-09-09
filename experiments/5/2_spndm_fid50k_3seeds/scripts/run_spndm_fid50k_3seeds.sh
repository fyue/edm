#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 5-2:
# S-PNDM sampler の FID-50k を3つの seed 範囲で評価します。

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"

EXP_ROOT="${EXP_ROOT:-experiments/5/2_spndm_fid50k_3seeds}"
CONFIG="${CONFIG:-$EXP_ROOT/config/spndm_fid50k_3seeds.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"
SEED_FILTER="${SEED_FILTER:-}"

NUM_IMAGES="${NUM_IMAGES:-50000}"
STEPS="${STEPS:-18}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SUMMARIZE="${SUMMARIZE:-1}"

MASTER_LOG="$LOG_DIR/spndm-fid50k-3seeds.log"
SUMMARY="$RESULTS_DIR/spndm_fid50k_3seeds_summary.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"

mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

is_url() {
    [[ "$1" =~ ^https?:// ]]
}

tokenize() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/\./p/g'
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'checkpoint\tlabel\tsampler\tS_churn\tnum_images\tseeds\tsteps\tS_min\tS_max\tS_noise\tfid_50k\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

condition_done() {
    local checkpoint="$1"
    local sampler="$2"
    local churn="$3"
    local seeds="$4"
    ensure_summary
    awk -F '\t' \
        -v checkpoint="$checkpoint" \
        -v sampler="$sampler" \
        -v churn="$churn" \
        -v num_images="$NUM_IMAGES" \
        -v seeds="$seeds" \
        -v steps="$STEPS" \
        'NR > 1 && $1 == checkpoint && $3 == sampler && $4 == churn && $5 == num_images && $6 == seeds && $7 == steps { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

matches_filter() {
    local value="$1"
    local filter="$2"
    local wanted
    if [[ -z "$filter" ]]; then
        return 0
    fi
    for wanted in $filter; do
        if [[ "$wanted" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

check_inputs() {
    local missing=0
    if [[ ! -f "$CONFIG" ]]; then
        log "Missing required file: $CONFIG"
        missing=1
    fi
    if [[ ! -f "$FID_REF" ]] && ! is_url "$FID_REF"; then
        log "Missing required file: $FID_REF"
        missing=1
    fi
    if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi
    if [[ "$SUMMARIZE" == "1" && ! -x "$PYTHON" ]]; then
        log "Missing executable python: $PYTHON"
        missing=1
    fi
    if (( missing != 0 )); then
        exit 1
    fi
}

count_pngs() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        printf '0'
        return
    fi
    find "$dir" -type f -name '*.png' | wc -l
}

run_condition() {
    local checkpoint="$1"
    local label="$2"
    local network="$3"
    local sampler="$4"
    local churn="$5"
    local seeds="$6"
    local sampler_token
    local churn_token
    local seeds_token
    local images
    local run_log
    local png_count
    local fid

    if [[ "$sampler" != "spndm" ]]; then
        log "[$checkpoint sampler=$sampler seeds=$seeds] Experiment 5-2 only accepts sampler=spndm"
        return 1
    fi

    if [[ ! -f "$network" ]] && ! is_url "$network"; then
        log "[$checkpoint sampler=$sampler seeds=$seeds] Missing network: $network"
        return 1
    fi

    if condition_done "$checkpoint" "$sampler" "$churn" "$seeds"; then
        log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Already recorded; skipping"
        return 0
    fi

    sampler_token="$(tokenize "$sampler")"
    churn_token="$(tokenize "$churn")"
    seeds_token="$(tokenize "$seeds")"
    images="$IMAGE_ROOT/$checkpoint/$sampler_token/S_churn-$churn_token/n${NUM_IMAGES}-seeds-$seeds_token"
    run_log="$LOG_DIR/fid50k-${checkpoint}-${sampler_token}-S_churn-${churn_token}-seeds-${seeds_token}.log"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] generate -> $images"
        log "PLAN [$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] fid --num=$NUM_IMAGES"
        return 0
    fi

    mkdir -p "$images"
    png_count="$(count_pngs "$images" | tr -d ' ')"

    if (( png_count >= NUM_IMAGES )); then
        log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Found $png_count pngs; skipping generation"
    else
        log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Generating $NUM_IMAGES images"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$network" \
            --outdir="$images" \
            --seeds="$seeds" \
            --subdirs \
            --batch="$GEN_BATCH" \
            --sampler="$sampler" \
            --steps="$STEPS" \
            --S_churn="$churn" \
            --S_min="$S_MIN" \
            --S_max="$S_MAX" \
            --S_noise="$S_NOISE" 2>&1 | tee -a "$run_log"
    fi

    log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Calculating FID-50k"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$checkpoint" "$label" "$sampler" "$churn" "$NUM_IMAGES" "$seeds" "$STEPS" \
        "$S_MIN" "$S_MAX" "$S_NOISE" "$fid" "$network" "$images" "$run_log" >> "$SUMMARY"
    log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Recorded FID-50k=$fid"
}

main() {
    check_inputs
    ensure_summary

    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Summary: $SUMMARY"
    log "Config: $CONFIG"
    log "NUM_IMAGES=$NUM_IMAGES STEPS=$STEPS"
    log "S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    local checkpoint
    local label
    local network
    local sampler
    local churn
    local seeds

    while IFS=$'\t' read -r checkpoint label network sampler churn seeds; do
        [[ -z "${checkpoint:-}" || "$checkpoint" =~ ^# ]] && continue
        if ! matches_filter "$checkpoint" "$CHECKPOINT_FILTER"; then
            log "[$checkpoint] Filtered out by CHECKPOINT_FILTER; skipping"
            continue
        fi
        if ! matches_filter "$seeds" "$SEED_FILTER"; then
            log "[$checkpoint sampler=$sampler S_churn=$churn seeds=$seeds] Filtered out by SEED_FILTER; skipping"
            continue
        fi
        run_condition "$checkpoint" "$label" "$network" "$sampler" "$churn" "$seeds"
        if [[ -f "$STOP_FILE" ]]; then
            log "Stop file found: $STOP_FILE"
            return 0
        fi
    done < "$CONFIG"

    if [[ "$PLAN_ONLY" != "1" && "$SUMMARIZE" == "1" ]]; then
        log "Summarizing 3 seed FID results"
        "$PYTHON" "$EXP_ROOT/scripts/summarize_spndm_fid50k_3seeds.py" \
            --summary "$SUMMARY" \
            --out "$RESULTS_DIR/spndm_fid50k_3seeds_stats.tsv" 2>&1 | tee -a "$MASTER_LOG"
    fi
}

main "$@"
