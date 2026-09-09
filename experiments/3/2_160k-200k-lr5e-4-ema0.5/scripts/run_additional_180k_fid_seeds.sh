#!/usr/bin/env bash
set -Eeuo pipefail

# Add the two missing 50k-seed FID evaluations for the 180k snapshot.
# The existing 0-49999 result is preserved and skipped.
#
# Plan only:
#   PLAN_ONLY=1 experiments/3/2_160k-200k-lr5e-4-ema0.5/scripts/run_additional_180k_fid_seeds.sh
#
# Run:
#   tmux new -s edm-exp3-fid180k
#   experiments/3/2_160k-200k-lr5e-4-ema0.5/scripts/run_additional_180k_fid_seeds.sh
#
# Progress:
#   tail -f experiments/3/2_160k-200k-lr5e-4-ema0.5/logs/fid-180000-additional-seeds.log

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/2_160k-200k-lr5e-4-ema0.5}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr5e-4-ema0.5-160k-200k}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr5e-4-ema0.5-160k-200k}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

TARGET_KIMG="${TARGET_KIMG:-180000}"
SNAPSHOT="${SNAPSHOT:-$RUN_DIR/network-snapshot-180000.pkl}"
ADDITIONAL_FID_SEEDS_LIST="${ADDITIONAL_FID_SEEDS_LIST:-50000-99999 100000-149999}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
LR="${LR:-0.0005}"
EMA="${EMA:-0.5}"
PLAN_ONLY="${PLAN_ONLY:-0}"

SUMMARY="$RESULTS_DIR/summary.tsv"
AGGREGATE="$RESULTS_DIR/fid-180000-aggregate.tsv"
MASTER_LOG="$LOG_DIR/fid-180000-additional-seeds.log"

mkdir -p "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

check_inputs() {
    local missing=0
    for path in "$SNAPSHOT" "$FID_REF" "$SUMMARY"; do
        if [[ ! -f "$path" ]]; then
            log "Missing required file: $path"
            missing=1
        fi
    done
    if (( missing != 0 )); then
        exit 1
    fi
}

fid_already_done() {
    local seeds="$1"
    awk -F '\t' -v target="$TARGET_KIMG" -v seeds="$seeds" \
        'NR > 1 && $1 == target && $2 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

run_fid_for() {
    local seeds="$1"
    local images="$FID_ROOT/images-${TARGET_KIMG}-${seeds}"
    local fid_log="$FID_ROOT/fid-${TARGET_KIMG}-${seeds}.log"
    local fid

    if fid_already_done "$seeds"; then
        log "FID already recorded for ${TARGET_KIMG} kimg seeds=$seeds; skipping"
        return 0
    fi

    log "Generating FID images for ${TARGET_KIMG} kimg seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$images" \
        --seeds="$seeds" \
        --subdirs \
        --batch="$GEN_BATCH" \
        --steps=18 \
        --network="$SNAPSHOT" 2>&1 | tee -a "$fid_log"

    log "Calculating FID for ${TARGET_KIMG} kimg seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "Could not parse FID from $fid_log"
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$TARGET_KIMG" "$seeds" "$fid" "$LR" "$EMA" "$SNAPSHOT" "$images" >> "$SUMMARY"
    log "Recorded FID=$fid for ${TARGET_KIMG} kimg seeds=$seeds"
}

write_aggregate() {
    awk -F '\t' -v target="$TARGET_KIMG" '
        BEGIN { OFS = "\t"; print "kimg", "num_seed_ranges", "fid_mean", "fid_std" }
        NR > 1 && $1 == target {
            sum += $3
            sumsq += $3 * $3
            count++
        }
        END {
            if (count == 0) exit 1
            mean = sum / count
            variance = sumsq / count - mean * mean
            if (variance < 0) variance = 0
            printf "%s\t%d\t%.5f\t%.5f\n", target, count, mean, sqrt(variance)
        }
    ' "$SUMMARY" > "$AGGREGATE"
    log "Aggregate result: $(tail -n 1 "$AGGREGATE")"
}

main() {
    check_inputs
    log "Snapshot: $SNAPSHOT"
    log "Additional FID seed ranges: $ADDITIONAL_FID_SEEDS_LIST"
    log "Summary: $SUMMARY"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no image generation or FID calculation will be executed"
        exit 0
    fi

    local seeds
    for seeds in $ADDITIONAL_FID_SEEDS_LIST; do
        run_fid_for "$seeds"
    done

    write_aggregate
    log "All additional 180k FID evaluations complete"
}

main "$@"
