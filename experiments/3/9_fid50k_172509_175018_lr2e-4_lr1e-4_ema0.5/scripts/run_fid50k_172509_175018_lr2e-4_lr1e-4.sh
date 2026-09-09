#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 3-9: evaluate lr=2e-4 and lr=1e-4 snapshots at 172509 and
# 175018 kimg over three independent FID-50k seed ranges. No training is run.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/3/9_fid50k_172509_175018_lr2e-4_lr1e-4_ema0.5}"
CONDITIONS="${CONDITIONS:-$EXP_ROOT/config/conditions.tsv}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
STEPS="${STEPS:-18}"
S_CHURN="${S_CHURN:-0}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fid50k-172509-175018-lr2e-4-lr1e-4.log"
SUMMARY="$RESULTS_DIR/summary.tsv"
AGGREGATE="$RESULTS_DIR/aggregate.tsv"

mkdir -p "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'condition\tlr\tema\tsnapshot_kimg\tseeds\tnum_images\tsteps\tS_churn\tfid\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

already_done() {
    local condition="$1"
    local seeds="$2"
    ensure_summary
    awk -F '\t' \
        -v condition="$condition" \
        -v seeds="$seeds" \
        -v num="$NUM_IMAGES" \
        -v steps="$STEPS" \
        -v churn="$S_CHURN" \
        'NR > 1 && $1 == condition && $5 == seeds && $6 == num && $7 == steps && $8 == churn { found = 1 } END { exit(found ? 0 : 1) }' \
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
    local condition lr ema snapshot_kimg network

    if [[ ! -f "$CONDITIONS" ]]; then
        log "Missing conditions file: $CONDITIONS"
        exit 1
    fi
    if [[ ! -f "$FID_REF" ]]; then
        log "Missing required FID reference: $FID_REF"
        missing=1
    fi
    if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi

    while IFS=$'\t' read -r condition lr ema snapshot_kimg network; do
        [[ "$condition" == "condition" ]] && continue
        [[ -z "${condition:-}" ]] && continue
        if [[ ! -f "$network" ]]; then
            log "Missing required snapshot for $condition: $network"
            missing=1
        fi
    done < "$CONDITIONS"

    if (( missing != 0 )) && [[ "$PLAN_ONLY" != "1" ]]; then
        exit 1
    elif (( missing != 0 )); then
        log "PLAN_ONLY=1; continuing despite missing inputs"
    fi
}

run_seed_range() {
    local condition="$1"
    local lr="$2"
    local ema="$3"
    local snapshot_kimg="$4"
    local network="$5"
    local seeds="$6"
    local image_dir="$FID_ROOT/$condition/images-${snapshot_kimg}-${seeds}"
    local fid_log="$FID_ROOT/$condition/fid-${snapshot_kimg}-${seeds}.log"
    local png_count
    local fid

    if already_done "$condition" "$seeds"; then
        log "[$condition seeds=$seeds] Matching FID result is already recorded; skipping"
        return 0
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition seeds=$seeds] generate $NUM_IMAGES images -> $image_dir"
        log "PLAN [$condition seeds=$seeds] calculate FID-${NUM_IMAGES}"
        return 0
    fi

    mkdir -p "$image_dir" "$(dirname "$fid_log")"
    png_count="$(count_pngs "$image_dir" | tr -d ' ')"
    if (( png_count >= NUM_IMAGES )); then
        log "[$condition seeds=$seeds] Found $png_count PNG files; skipping generation"
    else
        log "[$condition seeds=$seeds] Generating $NUM_IMAGES images"
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
        log "[$condition seeds=$seeds] Expected at least $NUM_IMAGES PNG files, found $png_count"
        return 1
    fi

    log "[$condition seeds=$seeds] Calculating FID-${NUM_IMAGES}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$image_dir" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$MASTER_LOG" "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[$condition seeds=$seeds] Could not parse FID from $fid_log"
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$condition" "$lr" "$ema" "$snapshot_kimg" "$seeds" "$NUM_IMAGES" "$STEPS" "$S_CHURN" "$fid" \
        "$network" "$image_dir" "$fid_log" >> "$SUMMARY"
    log "[$condition seeds=$seeds] Recorded FID-${NUM_IMAGES}=$fid in $SUMMARY"
}

write_aggregate() {
    ensure_summary
    awk -F '\t' '
        BEGIN {
            OFS = "\t"
            print "condition", "lr", "ema", "snapshot_kimg", "num_results", "fid_mean", "fid_std"
        }
        NR > 1 {
            key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
            n[key] += 1
            sum[key] += $9
            sumsq[key] += $9 * $9
            if (!(key in order_seen)) {
                order_seen[key] = 1
                order[++order_count] = key
                condition[key] = $1
                lr[key] = $2
                ema[key] = $3
                snapshot_kimg[key] = $4
            }
        }
        END {
            for (i = 1; i <= order_count; i++) {
                key = order[i]
                mean = sum[key] / n[key]
                variance = 0
                if (n[key] > 1) {
                    variance = (sumsq[key] - sum[key] * sum[key] / n[key]) / (n[key] - 1)
                    if (variance < 0 && variance > -1e-12) {
                        variance = 0
                    }
                }
                printf "%s\t%s\t%s\t%s\t%d\t%.6g\t%.6g\n", condition[key], lr[key], ema[key], snapshot_kimg[key], n[key], mean, sqrt(variance)
            }
        }
    ' "$SUMMARY" > "$AGGREGATE"
}

main() {
    local condition lr ema snapshot_kimg network seeds

    check_inputs
    ensure_summary

    log "Experiment 3-9: FID-${NUM_IMAGES} for lr=2e-4/lr=1e-4 snapshots 172509 and 175018"
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Conditions: $CONDITIONS"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "steps=$STEPS S_churn=$S_CHURN S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    while IFS=$'\t' read -r condition lr ema snapshot_kimg network; do
        [[ "$condition" == "condition" ]] && continue
        [[ -z "${condition:-}" ]] && continue
        for seeds in $FID_SEEDS_LIST; do
            run_seed_range "$condition" "$lr" "$ema" "$snapshot_kimg" "$network" "$seeds"
        done
    done < "$CONDITIONS"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; image generation and FID calculation were not executed"
    else
        write_aggregate
        log "All condition/seed ranges complete. Results: $SUMMARY"
        log "Aggregate: $AGGREGATE"
    fi
}

main "$@"
