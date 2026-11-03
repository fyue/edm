#!/usr/bin/env bash
set -Eeuo pipefail

# CIFAR-10 class-conditional DDPM++ EDM の 120k checkpoint を起点に、
# lr と EMA half-life だけを変えた 120k -> 160k 分岐実験を行うスクリプトです。
#
# 目的:
# - 120k 時点で FID が最良だったため、その先の FID 悪化が
#   学習率や EMA の影響を受けているかを確認します。
# - 先生からの提案に合わせて、以下の3条件を比較します。
#     1. lr=0.001,  ema=0.5  : 再現確認
#     2. lr=0.0005, ema=0.5  : learning rate を下げる
#     3. lr=0.001,  ema=1.0  : EMA half-life を長くする
# - 各条件は可能であれば3 seed範囲でFID評価します。
#
# 重要:
# - 削除処理は入れていません。既存ファイルは消しません。
# - 出力は experiments/3 配下に限定します。
# - 120k 起点ファイルは experiment1 の保存済みファイルを参照します。
# - 学習を途中で止めた場合、このスクリプトを再実行すると各分岐の
#   最新 training-state-*.pt から再開します。
# - 分岐ごとの 160k state が既にある場合、学習はスキップします。
# - FID が summary.tsv に記録済みの seed 範囲はスキップします。
#
# tmuxでの起動例:
#   tmux new -s edm-exp3
#   experiments/3/1_120k-160k-lr-ema-sweep/scripts/run_lr_ema_branches_from_120k.sh
#
# tmuxから切り離す:
#   Ctrl-b を押してから d
#
# tmuxに戻る:
#   tmux attach -t edm-exp3
#
# 進捗確認:
#   tail -f experiments/3/1_120k-160k-lr-ema-sweep/logs/lr-ema-branches.log
#
# 結果確認:
#   cat experiments/3/1_120k-160k-lr-ema-sweep/results/summary.tsv
#   cat experiments/3/1_120k-160k-lr-ema-sweep/results/health.tsv
#
# 実行前の設定確認だけを行う場合:
#   PLAN_ONLY=1 experiments/3/1_120k-160k-lr-ema-sweep/scripts/run_lr_ema_branches_from_120k.sh
#
# 現在の分岐が終わったところで止めたい場合:
#   touch experiments/3/1_120k-160k-lr-ema-sweep/STOP_AFTER_CURRENT_BRANCH

cd "$(dirname "$0")/../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"

EXP_ROOT="${EXP_ROOT:-experiments/3/1_120k-160k-lr-ema-sweep}"
CONFIG="${CONFIG:-$EXP_ROOT/config/branches.tsv}"
RUN_ROOT="${RUN_ROOT:-$EXP_ROOT/training-runs}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

BASE_RUN="${BASE_RUN:-experiments/1-cifar10-cond-ddpmpp-edm-seed0-200k-fid2.18/training-runs/cifar10-cond-ddpmpp-edm-managed}"
BASE_STATE="${BASE_STATE:-$BASE_RUN/training-state-120000.pt}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-$BASE_RUN/network-snapshot-120000.pkl}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"

START_KIMG="${START_KIMG:-120000}"
TARGET_KIMG="${TARGET_KIMG:-160000}"
TICK_KIMG="${TICK_KIMG:-50}"
SNAP_TICKS="${SNAP_TICKS:-50}"
DUMP_TICKS="${DUMP_TICKS:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
TRAIN_SEED="${TRAIN_SEED:-0}"

FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
FID_BATCH="${FID_BATCH:-64}"
GEN_BATCH="${GEN_BATCH:-64}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/lr-ema-branches.log"
SUMMARY="$RESULTS_DIR/summary.tsv"
HEALTH="$RESULTS_DIR/health.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_BRANCH"

mkdir -p "$RUN_ROOT" "$FID_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

kimg_to_mimg() {
    awk -v kimg="$1" 'BEGIN { printf "%.6g", kimg / 1000 }'
}

state_kimg() {
    basename "$1" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/' | sed 's/^0*//; s/^$/0/'
}

snapshot_path() {
    local run_dir="$1"
    local kimg="$2"
    printf '%s/network-snapshot-%06d.pkl' "$run_dir" "$kimg"
}

state_path() {
    local run_dir="$1"
    local kimg="$2"
    printf '%s/training-state-%06d.pt' "$run_dir" "$kimg"
}

latest_state() {
    local run_dir="$1"
    find "$run_dir" -maxdepth 1 -type f -name 'training-state-*.pt' 2>/dev/null | sort | tail -n 1
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'branch_id\tlabel\tlr\tema\tkimg\tseeds\tfid\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'branch_id\tlabel\tlr\tema\tkimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

fid_already_done() {
    local branch_id="$1"
    local seeds="$2"
    ensure_summary
    awk -F '\t' -v branch_id="$branch_id" -v target="$TARGET_KIMG" -v seeds="$seeds" \
        'NR > 1 && $1 == branch_id && $5 == target && $6 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

health_already_done() {
    local branch_id="$1"
    ensure_health
    awk -F '\t' -v branch_id="$branch_id" -v target="$TARGET_KIMG" \
        'NR > 1 && $1 == branch_id && $5 == target { found = 1 } END { exit(found ? 0 : 1) }' \
        "$HEALTH"
}

check_inputs() {
    local missing=0
    for path in "$CONFIG" "$BASE_STATE" "$BASE_SNAPSHOT" "$DATASET" "$FID_REF"; do
        if [[ ! -f "$path" ]]; then
            log "Missing required file: $path"
            missing=1
        fi
    done
    if (( missing != 0 )); then
        exit 1
    fi
}

run_train_branch() {
    local branch_id="$1"
    local label="$2"
    local lr="$3"
    local ema="$4"
    local run_dir="$RUN_ROOT/$branch_id"
    local duration
    local resume_state
    local current_kimg=0

    mkdir -p "$run_dir"
    duration="$(kimg_to_mimg "$TARGET_KIMG")"
    resume_state="$(latest_state "$run_dir" || true)"

    if [[ -n "$resume_state" ]]; then
        current_kimg="$(state_kimg "$resume_state")"
    else
        resume_state="$BASE_STATE"
        current_kimg="$START_KIMG"
    fi

    if (( current_kimg >= TARGET_KIMG )) && [[ -f "$(state_path "$run_dir" "$TARGET_KIMG")" ]]; then
        log "[$branch_id] Already have ${TARGET_KIMG} kimg state; skipping training"
        return 0
    fi

    log "[$branch_id] Training ${current_kimg} -> ${TARGET_KIMG} kimg; lr=$lr ema=$ema resume=$resume_state"
    "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        --outdir="$run_dir" --nosubdir \
        --data="$DATASET" \
        --cond=1 --arch=ddpmpp --precond=edm \
        --batch=512 --batch-gpu="$BATCH_GPU" \
        --lr="$lr" --ema="$ema" \
        --dropout=0.13 --augment=0.12 --fp16=0 \
        --seed="$TRAIN_SEED" \
        --tick="$TICK_KIMG" --snap="$SNAP_TICKS" --dump="$DUMP_TICKS" \
        --duration="$duration" \
        --resume="$resume_state" 2>&1 | tee -a "$MASTER_LOG"

    if [[ ! -f "$(state_path "$run_dir" "$TARGET_KIMG")" ]]; then
        log "[$branch_id] Expected final state not found: $(state_path "$run_dir" "$TARGET_KIMG")"
        exit 1
    fi
}

record_health() {
    local branch_id="$1"
    local label="$2"
    local lr="$3"
    local ema="$4"
    local run_dir="$RUN_ROOT/$branch_id"
    local state
    local snapshot
    local log_line
    local sec_per_kimg
    local gpumem
    local reserved
    local latest_loss
    local loss_status

    if health_already_done "$branch_id"; then
        log "[$branch_id] Health already recorded; skipping"
        return 0
    fi

    state="$(state_path "$run_dir" "$TARGET_KIMG")"
    snapshot="$(snapshot_path "$run_dir" "$TARGET_KIMG")"
    log_line="$(grep '^tick ' "$run_dir/log.txt" | tail -n 1 || true)"
    sec_per_kimg="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "sec/kimg") { print $(i + 1); exit } }' <<< "$log_line")"
    gpumem="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "gpumem") { print $(i + 1); exit } }' <<< "$log_line")"
    reserved="$(awk '{ for (i = 1; i <= NF; i++) if ($i == "reserved") { print $(i + 1); exit } }' <<< "$log_line")"

    latest_loss="$("$PYTHON" - "$run_dir/stats.jsonl" <<'PY'
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$branch_id" "$label" "$lr" "$ema" "$TARGET_KIMG" "${latest_loss:-NA}" "$loss_status" \
        "${sec_per_kimg:-NA}" "${gpumem:-NA}" "${reserved:-NA}" "$state" "$snapshot" >> "$HEALTH"

    log "[$branch_id] Recorded health: loss=${latest_loss:-NA} status=$loss_status sec/kimg=${sec_per_kimg:-NA}"
}

generate_samples() {
    local branch_id="$1"
    local snapshot="$RUN_ROOT/$branch_id/network-snapshot-$(printf '%06d' "$TARGET_KIMG").pkl"
    local sample_dir="$FID_ROOT/$branch_id/samples-${TARGET_KIMG}-${SAMPLE_SEEDS}"

    if [[ "$GENERATE_SAMPLES" != "1" ]]; then
        log "[$branch_id] Sample generation disabled"
        return 0
    fi
    if [[ -d "$sample_dir" ]]; then
        log "[$branch_id] Samples already exist; skipping"
        return 0
    fi

    mkdir -p "$FID_ROOT/$branch_id"
    log "[$branch_id] Generating samples seeds=$SAMPLE_SEEDS"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$sample_dir" \
        --seeds="$SAMPLE_SEEDS" \
        --batch="$GEN_BATCH" \
        --steps=18 \
        --network="$snapshot" 2>&1 | tee -a "$MASTER_LOG"
}

run_fid() {
    local branch_id="$1"
    local label="$2"
    local lr="$3"
    local ema="$4"
    local seeds="$5"
    local snapshot="$RUN_ROOT/$branch_id/network-snapshot-$(printf '%06d' "$TARGET_KIMG").pkl"
    local images="$FID_ROOT/$branch_id/images-${TARGET_KIMG}-${seeds}"
    local fid_log="$FID_ROOT/$branch_id/fid-${TARGET_KIMG}-${seeds}.log"
    local fid

    if fid_already_done "$branch_id" "$seeds"; then
        log "[$branch_id] FID already recorded for seeds=$seeds; skipping"
        return 0
    fi

    mkdir -p "$FID_ROOT/$branch_id"
    log "[$branch_id] Generating FID images seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$images" \
        --seeds="$seeds" \
        --subdirs \
        --batch="$GEN_BATCH" \
        --steps=18 \
        --network="$snapshot" 2>&1 | tee -a "$fid_log"

    log "[$branch_id] Calculating FID seeds=$seeds"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[$branch_id] Could not parse FID from $fid_log"
        exit 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$branch_id" "$label" "$lr" "$ema" "$TARGET_KIMG" "$seeds" "$fid" "$snapshot" "$images" >> "$SUMMARY"
    log "[$branch_id] Recorded FID=$fid seeds=$seeds"
}

run_branch() {
    local branch_id="$1"
    local label="$2"
    local lr="$3"
    local ema="$4"
    local seeds

    log "[$branch_id] ===== start label=$label lr=$lr ema=$ema ====="
    run_train_branch "$branch_id" "$label" "$lr" "$ema"
    record_health "$branch_id" "$label" "$lr" "$ema"
    generate_samples "$branch_id"

    for seeds in $FID_SEEDS_LIST; do
        run_fid "$branch_id" "$label" "$lr" "$ema" "$seeds"
    done

    log "[$branch_id] ===== complete ====="
}

print_plan() {
    local branch_id
    local label
    local lr
    local ema

    log "PLAN_ONLY=1; no training or FID commands will be executed"
    log "Planned branches:"
    while IFS=$'\t' read -r branch_id label lr ema; do
        if [[ "$branch_id" == "branch_id" || -z "$branch_id" ]]; then
            continue
        fi
        log "  branch=$branch_id label=$label lr=$lr ema=$ema outdir=$RUN_ROOT/$branch_id"
    done < "$CONFIG"
}

main() {
    check_inputs
    ensure_summary
    ensure_health

    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Experiment root: $EXP_ROOT"
    log "Base state: $BASE_STATE"
    log "Base snapshot: $BASE_SNAPSHOT"
    log "Target: ${START_KIMG} -> ${TARGET_KIMG} kimg"
    log "FID seed ranges: $FID_SEEDS_LIST"
    log "Stop file after current branch: $STOP_FILE"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        print_plan
        exit 0
    fi

    while IFS=$'\t' read -r branch_id label lr ema; do
        if [[ "$branch_id" == "branch_id" || -z "$branch_id" ]]; then
            continue
        fi
        run_branch "$branch_id" "$label" "$lr" "$ema"
        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after branch $branch_id"
            exit 0
        fi
    done < "$CONFIG"

    log "All branches complete"
    log "Summary:"
    cat "$SUMMARY" | tee -a "$MASTER_LOG"
}

main "$@"
