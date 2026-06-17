#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the best 120k -> 160k branch:
#   lr=0.0005, EMA half-life=0.5
# from 160k to 200k, with FID checks at 180k and 200k.
#
# tmux example:
#   tmux new -s edm-exp3-200k
#   experiments/3/160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh
#
# Progress:
#   tail -f experiments/3/160k-200k-lr5e-4-ema0.5/logs/train-eval.log
#
# Results:
#   cat experiments/3/160k-200k-lr5e-4-ema0.5/results/summary.tsv
#   cat experiments/3/160k-200k-lr5e-4-ema0.5/results/health.tsv
#
# Plan-only check:
#   PLAN_ONLY=1 experiments/3/160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh
#
# Safe pause after the current stage:
#   touch experiments/3/160k-200k-lr5e-4-ema0.5/STOP_AFTER_CURRENT_STAGE

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

EXP_ROOT="${EXP_ROOT:-experiments/3/160k-200k-lr5e-4-ema0.5}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr5e-4-ema0.5-160k-200k}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr5e-4-ema0.5-160k-200k}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

BASE_RUN="${BASE_RUN:-experiments/3/120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5}"
BASE_STATE="${BASE_STATE:-$BASE_RUN/training-state-160000.pt}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-$BASE_RUN/network-snapshot-160000.pkl}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"

START_KIMG="${START_KIMG:-160000}"
TARGET_KIMG="${TARGET_KIMG:-200000}"
EVAL_KIMGS="${EVAL_KIMGS:-180000 200000}"
LR="${LR:-0.0005}"
EMA="${EMA:-0.5}"
TICK_KIMG="${TICK_KIMG:-50}"
SNAP_TICKS="${SNAP_TICKS:-50}"
DUMP_TICKS="${DUMP_TICKS:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
TRAIN_SEED="${TRAIN_SEED:-0}"

INTERIM_FID_SEEDS_LIST="${INTERIM_FID_SEEDS_LIST:-0-49999}"
FINAL_FID_SEEDS_LIST="${FINAL_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
FID_BATCH="${FID_BATCH:-64}"
GEN_BATCH="${GEN_BATCH:-64}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/train-eval.log"
SUMMARY="$RESULTS_DIR/summary.tsv"
HEALTH="$RESULTS_DIR/health.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_STAGE"

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
        printf 'kimg\tseeds\tfid\tlr\tema\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'kimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

check_inputs() {
    local missing=0
    for path in "$BASE_STATE" "$BASE_SNAPSHOT" "$DATASET" "$FID_REF"; do
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
    local kimg="$1"
    local seeds="$2"
    ensure_summary
    awk -F '\t' -v kimg="$kimg" -v seeds="$seeds" \
        'NR > 1 && $1 == kimg && $2 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

health_already_done() {
    local kimg="$1"
    ensure_health
    awk -F '\t' -v kimg="$kimg" \
        'NR > 1 && $1 == kimg { found = 1 } END { exit(found ? 0 : 1) }' \
        "$HEALTH"
}

run_train_to() {
    local target="$1"
    local duration
    local resume_state
    local current_kimg=0

    duration="$(kimg_to_mimg "$target")"
    resume_state="$(latest_state || true)"
    if [[ -n "$resume_state" ]]; then
        current_kimg="$(state_kimg "$resume_state")"
    else
        resume_state="$BASE_STATE"
        current_kimg="$START_KIMG"
    fi

    if (( current_kimg >= target )) && [[ -f "$(state_path "$target")" ]]; then
        log "Already have ${target} kimg state; skipping training"
        return 0
    fi

    log "Training ${current_kimg} -> ${target} kimg; lr=$LR ema=$EMA resume=$resume_state"
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

    if [[ ! -f "$(state_path "$target")" ]]; then
        log "Expected state not found: $(state_path "$target")"
        exit 1
    fi
}

record_health_for() {
    local target="$1"
    local log_line
    local sec_per_kimg
    local gpumem
    local reserved
    local latest_loss
    local loss_status

    if health_already_done "$target"; then
        log "Health already recorded for ${target} kimg; skipping"
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

    ensure_health
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$target" "${latest_loss:-NA}" "$loss_status" "${sec_per_kimg:-NA}" \
        "${gpumem:-NA}" "${reserved:-NA}" "$(state_path "$target")" "$(snapshot_path "$target")" >> "$HEALTH"

    log "Recorded health for ${target} kimg: loss=${latest_loss:-NA} status=$loss_status sec/kimg=${sec_per_kimg:-NA}"
}

generate_samples_for() {
    local target="$1"
    local sample_dir="$FID_ROOT/samples-${target}-${SAMPLE_SEEDS}"

    if [[ "$GENERATE_SAMPLES" != "1" ]]; then
        log "Sample generation disabled for ${target} kimg"
        return 0
    fi
    if [[ -d "$sample_dir" ]]; then
        log "Samples already exist for ${target} kimg; skipping"
        return 0
    fi

    log "Generating samples for ${target} kimg seeds=$SAMPLE_SEEDS"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$sample_dir" \
        --seeds="$SAMPLE_SEEDS" \
        --batch="$GEN_BATCH" \
        --steps=18 \
        --network="$(snapshot_path "$target")" 2>&1 | tee -a "$MASTER_LOG"
}

run_fid_for() {
    local target="$1"
    local seeds="$2"
    local images="$FID_ROOT/images-${target}-${seeds}"
    local fid_log="$FID_ROOT/fid-${target}-${seeds}.log"
    local fid

    if fid_already_done "$target" "$seeds"; then
        log "FID already recorded for ${target} kimg seeds=$seeds; skipping"
        return 0
    fi

    log "Generating FID images for ${target} kimg seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$images" \
        --seeds="$seeds" \
        --subdirs \
        --batch="$GEN_BATCH" \
        --steps=18 \
        --network="$(snapshot_path "$target")" 2>&1 | tee -a "$fid_log"

    log "Calculating FID for ${target} kimg seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "Could not parse FID from $fid_log"
        exit 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$target" "$seeds" "$fid" "$LR" "$EMA" "$(snapshot_path "$target")" "$images" >> "$SUMMARY"
    log "Recorded FID=$fid for ${target} kimg seeds=$seeds"
}

seed_ranges_for() {
    local target="$1"
    if (( target == TARGET_KIMG )); then
        printf '%s\n' "$FINAL_FID_SEEDS_LIST"
    else
        printf '%s\n' "$INTERIM_FID_SEEDS_LIST"
    fi
}

print_plan() {
    log "PLAN_ONLY=1; no training or FID commands will be executed"
    log "Experiment root: $EXP_ROOT"
    log "Run dir: $RUN_DIR"
    log "Base state: $BASE_STATE"
    log "Base snapshot: $BASE_SNAPSHOT"
    log "Train: ${START_KIMG} -> ${TARGET_KIMG} kimg; lr=$LR ema=$EMA"
    log "Eval kimgs: $EVAL_KIMGS"
    log "Interim FID seeds: $INTERIM_FID_SEEDS_LIST"
    log "Final FID seeds: $FINAL_FID_SEEDS_LIST"
}

main() {
    check_inputs
    ensure_summary
    ensure_health

    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Experiment root: $EXP_ROOT"
    log "Base state: $BASE_STATE"
    log "Target: ${START_KIMG} -> ${TARGET_KIMG} kimg; lr=$LR ema=$EMA"
    log "Eval kimgs: $EVAL_KIMGS"
    log "Stop file after current stage: $STOP_FILE"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        print_plan
        exit 0
    fi

    local target
    local seeds
    for target in $EVAL_KIMGS; do
        if (( target > TARGET_KIMG )); then
            log "Skipping ${target} kimg because it is beyond TARGET_KIMG=${TARGET_KIMG}"
            continue
        fi

        run_train_to "$target"
        record_health_for "$target"
        generate_samples_for "$target"

        for seeds in $(seed_ranges_for "$target"); do
            run_fid_for "$target" "$seeds"
        done

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed stage ${target} kimg"
            exit 0
        fi
    done

    log "All stages complete"
    log "Summary:"
    cat "$SUMMARY" | tee -a "$MASTER_LOG"
}

main "$@"
