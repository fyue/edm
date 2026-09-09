#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 3-18: continue the 160k checkpoint whose 3-seed FID is about
# 2.05 to 175k with lr=1e-4/EMA=0.5. After training reaches 175k, evaluate
# every generated checkpoint in (160k, 175k] over three FID-50k seed ranges.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"

EXP_ROOT="${EXP_ROOT:-experiments/3/18_160k-fid2.05-to-175k-lr1e-4-ema0.5}"
CONFIG="${CONFIG:-$EXP_ROOT/config/experiment.env}"
if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
fi

RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr1e-4-ema0.5-160k-175k}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr1e-4-ema0.5-160k-175k}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

BASE_RUN="${BASE_RUN:-experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr1e-3-ema0.5-repro}"
BASE_STATE="${BASE_STATE:-$BASE_RUN/training-state-160000.pt}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-$BASE_RUN/network-snapshot-160000.pkl}"
DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"

START_KIMG="${START_KIMG:-160000}"
TARGET_KIMG="${TARGET_KIMG:-175000}"
LR="${LR:-0.0001}"
EMA="${EMA:-0.5}"
TICK_KIMG="${TICK_KIMG:-50}"
SNAP_TICKS="${SNAP_TICKS:-50}"
DUMP_TICKS="${DUMP_TICKS:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
TRAIN_SEED="${TRAIN_SEED:-0}"

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

MASTER_LOG="$LOG_DIR/train-then-eval-all.log"
SUMMARY="$RESULTS_DIR/summary.tsv"
AGGREGATE="$RESULTS_DIR/aggregate.tsv"
HEALTH="$RESULTS_DIR/health.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CHECKPOINT"

mkdir -p "$RUN_DIR" "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

kimg_to_mimg() {
    awk -v kimg="$1" 'BEGIN { printf "%.6g", kimg / 1000 }'
}

state_kimg() {
    basename "$1" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/' | sed 's/^0*//; s/^$/0/'
}

state_path() {
    local kimg="$1"
    printf '%s/training-state-%06d.pt' "$RUN_DIR" "$kimg"
}

snapshot_path() {
    local kimg="$1"
    printf '%s/network-snapshot-%06d.pkl' "$RUN_DIR" "$kimg"
}

latest_state() {
    find "$RUN_DIR" -maxdepth 1 -type f -name 'training-state-*.pt' 2>/dev/null | sort | tail -n 1
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'snapshot_kimg\tseeds\tnum_images\tsteps\tS_churn\tfid\tlr\tema\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'kimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

check_inputs() {
    local missing=0
    local path
    for path in "$BASE_STATE" "$BASE_SNAPSHOT" "$DATASET" "$FID_REF"; do
        if [[ ! -f "$path" ]]; then
            log "Missing required file: $path"
            missing=1
        fi
    done
    if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        log "Missing executable python: $PYTHON"
        missing=1
    fi
    if (( missing != 0 )) && [[ "$PLAN_ONLY" != "1" ]]; then
        exit 1
    fi
}

train_to_target() {
    local duration
    local resume_state
    local current_kimg

    duration="$(kimg_to_mimg "$TARGET_KIMG")"
    resume_state="$(latest_state || true)"
    if [[ -n "$resume_state" ]]; then
        current_kimg="$(state_kimg "$resume_state")"
    else
        resume_state="$BASE_STATE"
        current_kimg="$START_KIMG"
    fi

    if (( current_kimg >= TARGET_KIMG )) && [[ -f "$(state_path "$TARGET_KIMG")" ]]; then
        log "Already have ${TARGET_KIMG} kimg state; skipping training"
        return 0
    fi

    log "Training ${current_kimg} -> ${TARGET_KIMG} kimg; lr=$LR ema=$EMA resume=$resume_state"
    "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        --outdir="$RUN_DIR" --nosubdir \
        --data="$DATASET" \
        --cond=1 --arch=ddpmpp --precond=edm \
        --batch=512 --batch-gpu="$BATCH_GPU" \
        --lr="$LR" --ema="$EMA" \
        --dropout=0.13 --augment=0.12 --fp16=0 \
        --seed="$TRAIN_SEED" \
        --tick="$TICK_KIMG" --snap="$SNAP_TICKS" --dump="$DUMP_TICKS" \
        --duration="$duration" \
        --resume="$resume_state" 2>&1 | tee -a "$MASTER_LOG"

    if [[ ! -f "$(state_path "$TARGET_KIMG")" ]]; then
        log "Expected final state not found: $(state_path "$TARGET_KIMG")"
        exit 1
    fi
}

discover_snapshots() {
    find "$RUN_DIR" -maxdepth 1 -type f -name 'network-snapshot-*.pkl' \
        | sed -E 's/.*network-snapshot-([0-9]+)\.pkl$/\1/' \
        | sed 's/^0*//; s/^$/0/' \
        | awk -v start="$START_KIMG" -v target="$TARGET_KIMG" '$1 > start && $1 <= target { print $1 }' \
        | sort -n
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

record_health() {
    local target="$1"
    local log_line sec_per_kimg gpumem reserved latest_loss loss_status

    if awk -F '\t' -v kimg="$target" 'NR > 1 && $1 == kimg { found = 1 } END { exit(found ? 0 : 1) }' "$HEALTH"; then
        return 0
    fi

    log_line="$(grep '^tick ' "$RUN_DIR/log.txt" | tail -n 1 || true)"
    sec_per_kimg="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "sec/kimg") { print $(i + 1); exit } }' <<< "$log_line")"
    gpumem="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "gpumem") { print $(i + 1); exit } }' <<< "$log_line")"
    reserved="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "reserved") { print $(i + 1); exit } }' <<< "$log_line")"

    latest_loss="$("$PYTHON" - "$RUN_DIR/stats.jsonl" <<'PY'
import json
import math
import sys

value = ""
try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            loss = row.get("Loss/loss")
            if isinstance(loss, dict):
                loss = loss.get("mean", loss.get("value"))
            if loss is not None:
                value = float(loss)
except FileNotFoundError:
    pass

if value == "":
    print("")
elif math.isfinite(value):
    print(f"{value:.9g}")
else:
    print(str(value))
PY
)"

    if [[ -z "$latest_loss" ]]; then
        loss_status="missing"
    elif [[ "$latest_loss" == "nan" || "$latest_loss" == "inf" || "$latest_loss" == "-inf" ]]; then
        loss_status="bad"
    else
        loss_status="ok"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$target" "${latest_loss:-NA}" "$loss_status" "${sec_per_kimg:-NA}" \
        "${gpumem:-NA}" "${reserved:-NA}" "$(state_path "$target")" "$(snapshot_path "$target")" >> "$HEALTH"
}

run_fid_for() {
    local kimg="$1"
    local seeds="$2"
    local network image_dir fid_log png_count fid

    network="$(snapshot_path "$kimg")"
    image_dir="$FID_ROOT/images-${kimg}-${seeds}"
    fid_log="$FID_ROOT/fid-${kimg}-${seeds}.log"

    if already_done "$kimg" "$seeds"; then
        log "[snapshot=$kimg seeds=$seeds] Matching FID result is already recorded; skipping"
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

write_aggregate() {
    ensure_summary
    awk -F '\t' '
        BEGIN {
            OFS = "\t"
            print "snapshot_kimg", "num_results", "fid_mean", "fid_std", "fid_min", "fid_max"
        }
        NR > 1 {
            key = $1
            n[key] += 1
            sum[key] += $6
            sumsq[key] += $6 * $6
            if (!(key in min) || $6 < min[key]) min[key] = $6
            if (!(key in max) || $6 > max[key]) max[key] = $6
            if (!(key in seen)) {
                seen[key] = 1
                order[++order_count] = key
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
                printf "%s\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n", key, n[key], mean, sqrt(variance), min[key], max[key]
            }
        }
    ' "$SUMMARY" > "$AGGREGATE"
}

print_plan() {
    log "PLAN_ONLY=1; no training, image generation, or FID calculation will be executed"
    log "Experiment root: $EXP_ROOT"
    log "Run dir: $RUN_DIR"
    log "Base state: $BASE_STATE"
    log "Base snapshot: $BASE_SNAPSHOT"
    log "Train: ${START_KIMG} -> ${TARGET_KIMG} kimg; lr=$LR ema=$EMA"
    log "After training, evaluate every network-snapshot in $RUN_DIR with:"
    log "  snapshot_kimg > $START_KIMG and snapshot_kimg <= $TARGET_KIMG"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "Sampling: steps=$STEPS S_churn=$S_CHURN S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"
}

main() {
    local snapshots
    local kimg seeds

    check_inputs
    ensure_summary
    ensure_health

    log "Experiment 3-18: train 160k FID~2.05 checkpoint to 175k with lr=1e-4, then evaluate all generated checkpoints"
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        print_plan
        exit 0
    fi

    train_to_target
    record_health "$TARGET_KIMG"

    snapshots="$(discover_snapshots)"
    if [[ -z "$snapshots" ]]; then
        log "No snapshots found in ($START_KIMG, $TARGET_KIMG]: $RUN_DIR"
        exit 1
    fi

    log "Discovered snapshots for evaluation: $(tr '\n' ' ' <<< "$snapshots")"
    while read -r kimg; do
        [[ -z "$kimg" ]] && continue
        for seeds in $FID_SEEDS_LIST; do
            run_fid_for "$kimg" "$seeds"
        done
        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed checkpoint ${kimg}"
            write_aggregate
            exit 0
        fi
    done <<< "$snapshots"

    write_aggregate
    log "All checkpoint/seed ranges complete. Results: $SUMMARY"
    log "Aggregate: $AGGREGATE"
}

main "$@"
