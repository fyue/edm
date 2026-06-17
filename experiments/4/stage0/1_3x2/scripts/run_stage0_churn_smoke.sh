#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 4 Stage 0 3x2:
# 40k/120k/pretrained checkpoints に対して S_churn=0/20 を振り、
# 1024 samples の FID-1k で早めに傾向を確認するためのスモークテストです。
#
# 重要:
# - FID-1k は正式なFIDではありません。条件間の相対的な傾向を見る用途です。
# - 出力は experiments/4/stage0/3x2 配下に限定します。
# - summary.tsv に記録済みの条件は再実行時にスキップします。
# - 既存ファイルの削除処理は入れていません。
#
# 実行前の確認:
#   PLAN_ONLY=1 experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
#
# 実行:
#   experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh
#
# 一部だけ実行:
#   CHECKPOINT_FILTER="40k pretrained" CHURNS="0 20" experiments/4/stage0/3x2/scripts/run_stage0_churn_smoke.sh

cd "$(dirname "$0")/../../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/4/stage0/3x2}"
CONFIG="${CONFIG:-$EXP_ROOT/config/stage0_checkpoints.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
CHURNS="${CHURNS:-0 20}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-40k 120k pretrained}"

NUM_IMAGES="${NUM_IMAGES:-1024}"
SEEDS="${SEEDS:-0-1023}"
STEPS="${STEPS:-18}"
S_MIN="${S_MIN:-0}"
S_MAX="${S_MAX:-inf}"
S_NOISE="${S_NOISE:-1}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/stage0-churn-smoke.log"
SUMMARY="$RESULTS_DIR/stage0_summary.tsv"
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
        printf 'checkpoint\tlabel\tS_churn\tnum_images\tseeds\tsteps\tS_min\tS_max\tS_noise\tfid_1k\tnetwork\timages\tlog\n' > "$SUMMARY"
    fi
}

condition_done() {
    local checkpoint="$1"
    local churn="$2"
    ensure_summary
    awk -F '\t' \
        -v checkpoint="$checkpoint" \
        -v churn="$churn" \
        -v num_images="$NUM_IMAGES" \
        -v seeds="$SEEDS" \
        -v steps="$STEPS" \
        'NR > 1 && $1 == checkpoint && $3 == churn && $4 == num_images && $5 == seeds && $6 == steps { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

should_run_checkpoint() {
    local checkpoint="$1"
    local wanted
    if [[ -z "$CHECKPOINT_FILTER" ]]; then
        return 0
    fi
    for wanted in $CHECKPOINT_FILTER; do
        if [[ "$wanted" == "$checkpoint" ]]; then
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
    if [[ ! -x "$TORCHRUN" ]]; then
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
    local churn_token
    local seeds_token
    local images
    local run_log
    local png_count
    local fid

    if [[ ! -f "$network" ]] && ! is_url "$network"; then
        log "[$checkpoint S_churn=$churn] Missing network: $network"
        return 1
    fi

    if condition_done "$checkpoint" "$churn"; then
        log "[$checkpoint S_churn=$churn] Already recorded; skipping"
        return 0
    fi

    churn_token="$(tokenize "$churn")"
    seeds_token="$(tokenize "$SEEDS")"
    images="$IMAGE_ROOT/$checkpoint/S_churn-$churn_token/n${NUM_IMAGES}-seeds-$seeds_token"
    run_log="$LOG_DIR/stage0-${checkpoint}-S_churn-${churn_token}-n${NUM_IMAGES}.log"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$checkpoint S_churn=$churn] generate -> $images"
        log "PLAN [$checkpoint S_churn=$churn] fid --num=$NUM_IMAGES"
        return 0
    fi

    mkdir -p "$images"
    png_count="$(count_pngs "$images" | tr -d ' ')"

    if (( png_count >= NUM_IMAGES )); then
        log "[$checkpoint S_churn=$churn] Found $png_count pngs; skipping generation"
    else
        log "[$checkpoint S_churn=$churn] Generating $NUM_IMAGES images seeds=$SEEDS"
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --network="$network" \
            --outdir="$images" \
            --seeds="$SEEDS" \
            --subdirs \
            --batch="$GEN_BATCH" \
            --steps="$STEPS" \
            --S_churn="$churn" \
            --S_min="$S_MIN" \
            --S_max="$S_MAX" \
            --S_noise="$S_NOISE" 2>&1 | tee -a "$run_log"
    fi

    log "[$checkpoint S_churn=$churn] Calculating FID-1k num=$NUM_IMAGES"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$checkpoint S_churn=$churn] Could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$checkpoint" "$label" "$churn" "$NUM_IMAGES" "$SEEDS" "$STEPS" \
        "$S_MIN" "$S_MAX" "$S_NOISE" "$fid" "$network" "$images" "$run_log" >> "$SUMMARY"
    log "[$checkpoint S_churn=$churn] Recorded FID-1k=$fid"
}

main() {
    check_inputs
    ensure_summary

    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Summary: $SUMMARY"
    log "Config: $CONFIG"
    log "CHURNS=$CHURNS NUM_IMAGES=$NUM_IMAGES SEEDS=$SEEDS STEPS=$STEPS"
    log "S_min=$S_MIN S_max=$S_MAX S_noise=$S_NOISE"

    local checkpoint
    local label
    local network
    local churn

    while IFS=$'\t' read -r checkpoint label network; do
        [[ -z "${checkpoint:-}" || "$checkpoint" =~ ^# ]] && continue
        if ! should_run_checkpoint "$checkpoint"; then
            log "[$checkpoint] Filtered out; skipping"
            continue
        fi
        for churn in $CHURNS; do
            run_condition "$checkpoint" "$label" "$network" "$churn"
            if [[ -f "$STOP_FILE" ]]; then
                log "Stop file found: $STOP_FILE"
                return 0
            fi
        done
    done < "$CONFIG"

    log "Done. Results: $SUMMARY"
}

main "$@"
