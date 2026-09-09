#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the saved ~175k snapshot from Experiment 3_5 over three
# independent 50k seed ranges. No training is performed.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/5_170k-200k-lr2e-4-ema0.5}"
NETWORK="${NETWORK:-$EXP_ROOT/training-runs/lr2e-4-ema0.5-170k-200k/network-snapshot-175018.pkl}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr2e-4-ema0.5-170k-200k}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

SNAPSHOT_KIMG="${SNAPSHOT_KIMG:-175018}"
FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
STEPS="${STEPS:-18}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
LR="${LR:-0.0002}"
EMA="${EMA:-0.5}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fid-175018.log"
SUMMARY="$RESULTS_DIR/summary.tsv"

mkdir -p "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'kimg\tseeds\tfid\tlr\tema\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

already_done() {
    local seeds="$1"
    ensure_summary
    awk -F '\t' -v kimg="$SNAPSHOT_KIMG" -v seeds="$seeds" \
        'NR > 1 && $1 == kimg && $2 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

count_pngs() {
    local image_dir="$1"
    if [[ ! -d "$image_dir" ]]; then
        printf '0'
        return
    fi
    find "$image_dir" -type f -name '*.png' | wc -l
}

check_inputs() {
    local missing=0
    for path in "$NETWORK" "$FID_REF"; do
        if [[ ! -f "$path" ]]; then
            log "Missing required file: $path"
            missing=1
        fi
    done
    if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi
    if (( missing != 0 )); then
        exit 1
    fi
}

run_seed_range() {
    local seeds="$1"
    local image_dir="$FID_ROOT/images-${SNAPSHOT_KIMG}-${seeds}"
    local fid_log="$FID_ROOT/fid-${SNAPSHOT_KIMG}-${seeds}.log"
    local png_count
    local fid

    if already_done "$seeds"; then
        log "FID already recorded for ${SNAPSHOT_KIMG} kimg seeds=$seeds; skipping"
        return 0
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN seeds=$seeds: generate images in $image_dir, then calculate FID"
        return 0
    fi

    mkdir -p "$image_dir"
    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count >= 50000 )); then
        log "Found $png_count PNG files for seeds=$seeds; skipping generation"
    else
        log "Generating FID images for ${SNAPSHOT_KIMG} kimg seeds=$seeds"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$NETWORK" \
            --outdir="$image_dir" \
            --seeds="$seeds" \
            --subdirs \
            --batch="$GEN_BATCH" \
            --steps="$STEPS" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"
    fi

    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count < 50000 )); then
        log "Expected 50000 PNG files for seeds=$seeds, found $png_count"
        return 1
    fi

    log "Calculating FID for ${SNAPSHOT_KIMG} kimg seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$image_dir" \
        --ref="$FID_REF" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "Could not parse FID from $fid_log"
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$SNAPSHOT_KIMG" "$seeds" "$fid" "$LR" "$EMA" "$NETWORK" "$image_dir" >> "$SUMMARY"
    log "Recorded FID=$fid for ${SNAPSHOT_KIMG} kimg seeds=$seeds"
}

main() {
    local seeds

    check_inputs
    ensure_summary
    log "Three-seed FID evaluation for requested ~175k checkpoint (snapshot ${SNAPSHOT_KIMG})"
    log "Network: $NETWORK"
    log "Seed ranges: $FID_SEEDS_LIST"

    for seeds in $FID_SEEDS_LIST; do
        run_seed_range "$seeds"
    done

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no generation or FID calculation was executed"
    else
        log "All 175k seed ranges complete"
    fi
}

main "$@"
