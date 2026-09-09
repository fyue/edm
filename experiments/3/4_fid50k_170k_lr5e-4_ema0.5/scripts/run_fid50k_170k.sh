#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 3-4: calculate FID-50k for the ~170k snapshot from the
# lr=0.0005, EMA=0.5 continuation run.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/4_fid50k_170k_lr5e-4_ema0.5}"
NETWORK="${NETWORK:-experiments/3/160k-200k-lr5e-4-ema0.5/training-runs/lr5e-4-ema0.5-160k-200k/network-snapshot-170035.pkl}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

REQUESTED_KIMG="${REQUESTED_KIMG:-170000}"
SNAPSHOT_KIMG="${SNAPSHOT_KIMG:-170035}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
STEPS="${STEPS:-18}"
S_CHURN="${S_CHURN:-0}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
LR="${LR:-0.0005}"
EMA="${EMA:-0.5}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fid50k-170k.log"
SUMMARY="$RESULTS_DIR/summary.tsv"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'requested_kimg\tsnapshot_kimg\tseeds\tnum_images\tsteps\tS_churn\tfid\tlr\tema\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

already_done() {
    local seeds="$1"
    ensure_summary
    awk -F '\t' \
        -v requested="$REQUESTED_KIMG" \
        -v snapshot="$SNAPSHOT_KIMG" \
        -v seeds="$seeds" \
        -v num="$NUM_IMAGES" \
        -v steps="$STEPS" \
        -v churn="$S_CHURN" \
        'NR > 1 && $1 == requested && $2 == snapshot && $3 == seeds && $4 == num && $5 == steps && $6 == churn { found = 1 } END { exit(found ? 0 : 1) }' \
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
    local image_dir="$IMAGE_ROOT/170k-seeds-$seeds"
    local fid_log="$LOG_DIR/fid50k-170k-seeds-$seeds.log"
    local png_count
    local fid

    if already_done "$seeds"; then
        log "[seeds=$seeds] Matching FID result is already recorded; skipping"
        return 0
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [seeds=$seeds] generate $NUM_IMAGES images -> $image_dir"
        log "PLAN [seeds=$seeds] calculate FID-${NUM_IMAGES}"
        return 0
    fi

    mkdir -p "$image_dir"
    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count >= NUM_IMAGES )); then
        log "[seeds=$seeds] Found $png_count PNG files; skipping generation"
    else
        log "[seeds=$seeds] Generating $NUM_IMAGES images"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$NETWORK" \
            --outdir="$image_dir" \
            --seeds="$seeds" \
            --subdirs \
            --batch="$GEN_BATCH" \
            --steps="$STEPS" \
            --S_churn="$S_CHURN" \
            --S_min="$S_MIN" \
            --S_max="$S_MAX" \
            --S_noise="$S_NOISE" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"
    fi

    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count < NUM_IMAGES )); then
        log "[seeds=$seeds] Expected at least $NUM_IMAGES PNG files, found $png_count"
        return 1
    fi

    log "[seeds=$seeds] Calculating FID-${NUM_IMAGES}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$image_dir" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[seeds=$seeds] Could not parse FID from $fid_log"
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$REQUESTED_KIMG" "$SNAPSHOT_KIMG" "$seeds" "$NUM_IMAGES" "$STEPS" "$S_CHURN" \
        "$fid" "$LR" "$EMA" "$NETWORK" "$image_dir" "$fid_log" >> "$SUMMARY"
    log "[seeds=$seeds] Recorded FID-${NUM_IMAGES}=$fid in $SUMMARY"
}

main() {
    local seeds

    check_inputs
    ensure_summary

    log "Experiment 3-4: FID-${NUM_IMAGES} for requested ${REQUESTED_KIMG} kimg (snapshot ${SNAPSHOT_KIMG})"
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Network: $NETWORK"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "steps=$STEPS S_churn=$S_CHURN S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    for seeds in $FID_SEEDS_LIST; do
        run_seed_range "$seeds"
    done

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; image generation and FID calculation were not executed"
    else
        log "All seed ranges complete. Results: $SUMMARY"
    fi
}

main "$@"
