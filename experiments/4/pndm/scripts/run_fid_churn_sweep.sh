#!/usr/bin/env bash
set -Eeuo pipefail

# Shared runner for Experiment 4 PNDM S_churn sweeps.
# Config columns:
# checkpoint<TAB>label<TAB>network<TAB>churns<TAB>seeds(optional)

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"
SAMPLER="pndm"

: "${EXP_ROOT:?EXP_ROOT must be set by the stage wrapper}"
: "${CONFIG:?CONFIG must be set by the stage wrapper}"
: "${NUM_IMAGES:?NUM_IMAGES must be set by the stage wrapper}"
: "${DEFAULT_SEEDS:?DEFAULT_SEEDS must be set by the stage wrapper}"

IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
SUMMARY="${SUMMARY:-$RESULTS_DIR/fid_summary.tsv}"
MASTER_LOG="${MASTER_LOG:-$LOG_DIR/run.log}"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"
CHURN_FILTER="${CHURN_FILTER:-}"
SEED_FILTER="${SEED_FILTER:-}"
STEPS="${STEPS:-18}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
PLAN_ONLY="${PLAN_ONLY:-0}"

log() {
    local message
    message="$(printf '[%(%Y-%m-%d %H:%M:%S)T] %s' -1 "$*")"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf '%s\n' "$message"
    else
        printf '%s\n' "$message" | tee -a "$MASTER_LOG"
    fi
}

is_url() {
    [[ "$1" =~ ^https?:// ]]
}

tokenize() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/\./p/g'
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

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'checkpoint\tlabel\tsampler\tS_churn\tnum_images\tseeds\tsteps\tS_min\tS_max\tS_noise\tfid\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

condition_done() {
    local checkpoint="$1"
    local churn="$2"
    local seeds="$3"
    [[ -f "$SUMMARY" ]] || return 1
    awk -F '\t' \
        -v checkpoint="$checkpoint" \
        -v sampler="$SAMPLER" \
        -v churn="$churn" \
        -v num_images="$NUM_IMAGES" \
        -v seeds="$seeds" \
        -v steps="$STEPS" \
        'NR > 1 && $1 == checkpoint && $3 == sampler && $4 == churn && $5 == num_images && $6 == seeds && $7 == steps { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
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
    local churn="$4"
    local seeds="$5"
    local churn_token
    local seeds_token
    local images
    local run_log
    local png_count
    local fid

    if [[ ! -f "$network" ]] && ! is_url "$network"; then
        log "[$checkpoint S_churn=$churn seeds=$seeds] Missing network: $network"
        return 1
    fi
    if condition_done "$checkpoint" "$churn" "$seeds"; then
        log "[$checkpoint S_churn=$churn seeds=$seeds] Already recorded; skipping"
        return 0
    fi

    churn_token="$(tokenize "$churn")"
    seeds_token="$(tokenize "$seeds")"
    images="$IMAGE_ROOT/$checkpoint/$SAMPLER/S_churn-$churn_token/n${NUM_IMAGES}-seeds-$seeds_token"
    run_log="$LOG_DIR/${checkpoint}-${SAMPLER}-S_churn-${churn_token}-seeds-${seeds_token}.log"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$checkpoint sampler=$SAMPLER S_churn=$churn seeds=$seeds] generate -> $images"
        log "PLAN [$checkpoint sampler=$SAMPLER S_churn=$churn seeds=$seeds] fid --num=$NUM_IMAGES"
        return 0
    fi

    mkdir -p "$images"
    png_count="$(count_pngs "$images" | tr -d ' ')"
    if (( png_count >= NUM_IMAGES )); then
        log "[$checkpoint S_churn=$churn seeds=$seeds] Found $png_count pngs; skipping generation"
    else
        log "[$checkpoint S_churn=$churn seeds=$seeds] Generating $NUM_IMAGES images"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$network" \
            --outdir="$images" \
            --seeds="$seeds" \
            --subdirs \
            --batch="$GEN_BATCH" \
            --sampler="$SAMPLER" \
            --steps="$STEPS" \
            --S_churn="$churn" \
            --S_min="$S_MIN" \
            --S_max="$S_MAX" \
            --S_noise="$S_NOISE" 2>&1 | tee -a "$run_log"
    fi

    log "[$checkpoint S_churn=$churn seeds=$seeds] Calculating FID num=$NUM_IMAGES"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$checkpoint S_churn=$churn seeds=$seeds] Could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$checkpoint" "$label" "$SAMPLER" "$churn" "$NUM_IMAGES" "$seeds" "$STEPS" \
        "$S_MIN" "$S_MAX" "$S_NOISE" "$fid" "$network" "$images" "$run_log" >> "$SUMMARY"
    log "[$checkpoint S_churn=$churn seeds=$seeds] Recorded FID=$fid"
}

main() {
    if [[ "$PLAN_ONLY" != "1" ]]; then
        mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"
    fi
    check_inputs
    if [[ "$PLAN_ONLY" != "1" ]]; then
        ensure_summary
    fi

    log "Config: $CONFIG"
    log "Summary: $SUMMARY"
    log "sampler=$SAMPLER NUM_IMAGES=$NUM_IMAGES STEPS=$STEPS"
    log "S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    local checkpoint
    local label
    local network
    local churns
    local row_seeds
    local seeds
    local churn

    while IFS=$'\t' read -r checkpoint label network churns row_seeds; do
        [[ -z "${checkpoint:-}" || "$checkpoint" =~ ^# ]] && continue
        seeds="${row_seeds:-$DEFAULT_SEEDS}"
        matches_filter "$checkpoint" "$CHECKPOINT_FILTER" || continue
        matches_filter "$seeds" "$SEED_FILTER" || continue
        for churn in $churns; do
            matches_filter "$churn" "$CHURN_FILTER" || continue
            run_condition "$checkpoint" "$label" "$network" "$churn" "$seeds"
            if [[ -f "$STOP_FILE" ]]; then
                log "Stop file found: $STOP_FILE"
                return 0
            fi
        done
    done < "$CONFIG"

    log "Done"
}

main "$@"
