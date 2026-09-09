#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 3-11: evaluate four lr=5e-4/EMA=0.5 checkpoints around 170k
# over three FID-50k seed ranges.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/11_fid50k_around170k_lr5e-4_ema0.5}"
CONFIG="${CONFIG:-$EXP_ROOT/config/experiment.env}"
if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

CHECKPOINTS_TSV="${CHECKPOINTS_TSV:-$EXP_ROOT/config/checkpoints.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
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

MASTER_LOG="$LOG_DIR/fid50k-around170k.log"
SUMMARY="$RESULTS_DIR/summary.tsv"
AGGREGATE="$RESULTS_DIR/aggregate.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CHECKPOINT"

mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'label\trequested_kimg\tsnapshot_kimg\tseeds\tnum_images\tsteps\tS_churn\tfid\tlr\tema\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

check_inputs() {
    local missing=0
    local line_no=0
    local label requested snapshot network

    if [[ ! -f "$CHECKPOINTS_TSV" ]]; then
        log "Missing checkpoint config: $CHECKPOINTS_TSV"
        missing=1
    fi
    if [[ ! -f "$FID_REF" ]]; then
        log "Missing FID reference: $FID_REF"
        missing=1
    fi
    if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi

    if [[ -f "$CHECKPOINTS_TSV" ]]; then
        while IFS=$'\t' read -r label requested snapshot network; do
            line_no=$((line_no + 1))
            if (( line_no == 1 )) || [[ -z "${label:-}" ]]; then
                continue
            fi
            if [[ -z "${network:-}" || ! -f "$network" ]]; then
                log "Missing network for checkpoint ${label:-line $line_no}: ${network:-<empty>}"
                missing=1
            fi
        done < "$CHECKPOINTS_TSV"
    fi

    if (( missing != 0 )) && [[ "$PLAN_ONLY" != "1" ]]; then
        exit 1
    fi
}

already_done() {
    local label="$1"
    local snapshot="$2"
    local seeds="$3"
    ensure_summary
    awk -F '\t' \
        -v label="$label" \
        -v snapshot="$snapshot" \
        -v seeds="$seeds" \
        -v num="$NUM_IMAGES" \
        -v steps="$STEPS" \
        -v churn="$S_CHURN" \
        'NR > 1 && $1 == label && $3 == snapshot && $4 == seeds && $5 == num && $6 == steps && $7 == churn { found = 1 } END { exit(found ? 0 : 1) }' \
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

run_fid_for() {
    local label="$1"
    local requested="$2"
    local snapshot="$3"
    local network="$4"
    local seeds="$5"
    local image_dir="$IMAGE_ROOT/${label}-snapshot-${snapshot}-seeds-${seeds}"
    local fid_log="$LOG_DIR/fid50k-${label}-snapshot-${snapshot}-seeds-${seeds}.log"
    local png_count
    local fid

    if already_done "$label" "$snapshot" "$seeds"; then
        log "[label=$label snapshot=$snapshot seeds=$seeds] Matching FID result is already recorded; skipping"
        return 0
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [label=$label snapshot=$snapshot seeds=$seeds] generate $NUM_IMAGES images -> $image_dir"
        log "PLAN [label=$label snapshot=$snapshot seeds=$seeds] calculate FID-${NUM_IMAGES}"
        return 0
    fi

    mkdir -p "$image_dir"
    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count >= NUM_IMAGES )); then
        log "[label=$label snapshot=$snapshot seeds=$seeds] Found $png_count PNG files; skipping generation"
    else
        log "[label=$label snapshot=$snapshot seeds=$seeds] Generating $NUM_IMAGES images"
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
        log "[label=$label snapshot=$snapshot seeds=$seeds] Expected at least $NUM_IMAGES PNG files, found $png_count"
        return 1
    fi

    log "[label=$label snapshot=$snapshot seeds=$seeds] Calculating FID-${NUM_IMAGES}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$image_dir" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[label=$label snapshot=$snapshot seeds=$seeds] Could not parse FID from $fid_log"
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$requested" "$snapshot" "$seeds" "$NUM_IMAGES" "$STEPS" "$S_CHURN" \
        "$fid" "$LR" "$EMA" "$network" "$image_dir" "$fid_log" >> "$SUMMARY"
    log "[label=$label snapshot=$snapshot seeds=$seeds] Recorded FID-${NUM_IMAGES}=$fid in $SUMMARY"
}

write_aggregate() {
    ensure_summary
    awk -F '\t' '
        BEGIN {
            OFS = "\t"
            print "label", "requested_kimg", "snapshot_kimg", "num_results", "fid_mean", "fid_std", "fid_min", "fid_max"
        }
        NR > 1 {
            key = $1 SUBSEP $2 SUBSEP $3
            n[key] += 1
            sum[key] += $8
            sumsq[key] += $8 * $8
            if (!(key in min) || $8 < min[key]) min[key] = $8
            if (!(key in max) || $8 > max[key]) max[key] = $8
            if (!(key in seen)) {
                seen[key] = 1
                order[++order_count] = key
                labels[key] = $1
                requested[key] = $2
                snapshot[key] = $3
            }
        }
        END {
            for (i = 1; i <= order_count; i++) {
                key = order[i]
                mean = sum[key] / n[key]
                variance = 0
                if (n[key] > 1) {
                    variance = (sumsq[key] - sum[key] * sum[key] / n[key]) / (n[key] - 1)
                    if (variance < 0 && variance > -1e-12) variance = 0
                }
                printf "%s\t%s\t%s\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n", labels[key], requested[key], snapshot[key], n[key], mean, sqrt(variance), min[key], max[key]
            }
        }
    ' "$SUMMARY" > "$AGGREGATE"
}

main() {
    local line_no=0
    local label requested snapshot network seeds

    check_inputs
    ensure_summary

    log "Experiment 3-11: FID-${NUM_IMAGES} for four lr=$LR EMA=$EMA checkpoints around 170k"
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Checkpoint config: $CHECKPOINTS_TSV"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "steps=$STEPS S_churn=$S_CHURN S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    while IFS=$'\t' read -r label requested snapshot network; do
        line_no=$((line_no + 1))
        if (( line_no == 1 )) || [[ -z "${label:-}" ]]; then
            continue
        fi
        log "Checkpoint: label=$label requested=${requested} snapshot=${snapshot} network=$network"
        for seeds in $FID_SEEDS_LIST; do
            run_fid_for "$label" "$requested" "$snapshot" "$network" "$seeds"
        done
        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed checkpoint ${label} (${snapshot})"
            write_aggregate
            exit 0
        fi
    done < "$CHECKPOINTS_TSV"

    write_aggregate
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; image generation and FID calculation were not executed"
    else
        log "All checkpoint/seed ranges complete. Results: $SUMMARY"
        log "Aggregate: $AGGREGATE"
    fi
}

main "$@"
