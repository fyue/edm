#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 3-8: evaluate two saved snapshots from Experiment 3-3 over
# three independent FID-50k seed ranges. No training is performed.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/8_fid50k_167526_165018_lr2e-4_ema0.5}"
CONFIG="${CONFIG:-$EXP_ROOT/config/experiment.env}"
if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

SOURCE_RUN="${SOURCE_RUN:-experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr2e-4-ema0.5-160k-170k}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

SNAPSHOT_KIMGS="${SNAPSHOT_KIMGS:-167526 165018}"
FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
STEPS="${STEPS:-18}"
S_CHURN="${S_CHURN:-0}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
LR="${LR:-0.0002}"
EMA="${EMA:-0.5}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fid50k-167526-165018.log"
SUMMARY="$RESULTS_DIR/summary.tsv"

mkdir -p "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

snapshot_path() {
    local kimg="$1"
    printf '%s/network-snapshot-%06d.pkl' "$SOURCE_RUN" "$kimg"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'snapshot_kimg\tseeds\tnum_images\tsteps\tS_churn\tfid\tlr\tema\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

already_done() {
    local kimg="$1"
    local seeds="$2"
    ensure_summary
    awk -F '\t' \
        -v kimg="$kimg" \
        -v seeds="$seeds" \
        -v num="$NUM_IMAGES" \
        -v steps="$STEPS" \
        -v churn="$S_CHURN" \
        'NR > 1 && $1 == kimg && $2 == seeds && $3 == num && $4 == steps && $5 == churn { found = 1 } END { exit(found ? 0 : 1) }' \
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
    local kimg

    for kimg in $SNAPSHOT_KIMGS; do
        local network
        network="$(snapshot_path "$kimg")"
        if [[ ! -f "$network" ]]; then
            log "Missing required snapshot: $network"
            missing=1
        fi
    done
    if [[ ! -f "$FID_REF" ]]; then
        log "Missing required FID reference: $FID_REF"
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

run_seed_range() {
    local kimg="$1"
    local seeds="$2"
    local network
    local image_dir
    local fid_log
    local png_count
    local fid

    network="$(snapshot_path "$kimg")"
    image_dir="$FID_ROOT/images-${kimg}-${seeds}"
    fid_log="$FID_ROOT/fid-${kimg}-${seeds}.log"

    if already_done "$kimg" "$seeds"; then
        log "[snapshot=$kimg seeds=$seeds] Matching FID result is already recorded; skipping"
        return 0
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [snapshot=$kimg seeds=$seeds] generate $NUM_IMAGES images -> $image_dir"
        log "PLAN [snapshot=$kimg seeds=$seeds] calculate FID-${NUM_IMAGES}"
        return 0
    fi

    mkdir -p "$image_dir"
    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count >= NUM_IMAGES )); then
        log "[snapshot=$kimg seeds=$seeds] Found $png_count PNG files; skipping generation"
    else
        log "[snapshot=$kimg seeds=$seeds] Generating $NUM_IMAGES images"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$network" \
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
        log "[snapshot=$kimg seeds=$seeds] Expected at least $NUM_IMAGES PNG files, found $png_count"
        return 1
    fi

    log "[snapshot=$kimg seeds=$seeds] Calculating FID-${NUM_IMAGES}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$image_dir" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[snapshot=$kimg seeds=$seeds] Could not parse FID from $fid_log"
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$kimg" "$seeds" "$NUM_IMAGES" "$STEPS" "$S_CHURN" "$fid" "$LR" "$EMA" \
        "$network" "$image_dir" "$fid_log" >> "$SUMMARY"
    log "[snapshot=$kimg seeds=$seeds] Recorded FID-${NUM_IMAGES}=$fid in $SUMMARY"
}

main() {
    local kimg
    local seeds

    check_inputs
    ensure_summary

    log "Experiment 3-8: FID-${NUM_IMAGES} for snapshots: $SNAPSHOT_KIMGS"
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Source run: $SOURCE_RUN"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "steps=$STEPS S_churn=$S_CHURN S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    for kimg in $SNAPSHOT_KIMGS; do
        for seeds in $FID_SEEDS_LIST; do
            run_seed_range "$kimg" "$seeds"
        done
    done

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; image generation and FID calculation were not executed"
    else
        log "All snapshot/seed ranges complete. Results: $SUMMARY"
    fi
}

main "$@"
