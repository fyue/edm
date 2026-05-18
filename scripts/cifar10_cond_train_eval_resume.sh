#!/usr/bin/env bash
set -Eeuo pipefail

# EDM CIFAR-10 class-conditional DDPM++ を、論文の FID 1.79 設定に
# できるだけ寄せて学習するための管理スクリプトです。
#
# 重要:
# - torchrun は SIGHUP で終了することがあるため、nohup ではなく tmux 推奨です。
# - 1 GPU環境なので、学習中に同時FID評価はせず、指定kimgごとに
#   学習プロセスを正常終了 -> FID評価 -> --resume で再開します。
# - いつ止まっても、このスクリプトを再実行すれば最新の training-state-*.pt
#   から再開し、記録済みFID評価はスキップします。
#
# tmuxでの起動:
#   tmux new -s edm
#   scripts/cifar10_cond_train_eval_resume.sh
#
# tmuxから切り離す:
#   Ctrl-b を押してから d
#
# tmuxに戻る:
#   tmux attach -t edm
#
# 進捗確認:
#   tail -f logs/cifar10-cond-train-eval-resume.log
#
# FID結果:
#   cat fid-evals/cifar10-cond-ddpmpp-edm-managed/summary.tsv
#
# 健全性確認:
#   cat fid-evals/cifar10-cond-ddpmpp-edm-managed/health.tsv
#
# 途中サンプル画像:
#   fid-evals/cifar10-cond-ddpmpp-edm-managed/samples-<kimg>-<seeds>/
#
# 現在の学習/評価ステージが終わったら安全に止めたい場合:
#   touch training-runs/cifar10-cond-ddpmpp-edm-managed/STOP_AFTER_CURRENT_STAGE
#
# 論文設定との対応:
# - READMEの CIFAR-10 class-conditional VP 設定は:
#     --cond=1 --arch=ddpmpp
# - train.py のデフォルトで:
#     --precond=edm --duration=200 --batch=512 --lr=0.001
#     --ema=0.5 --dropout=0.13 --augment=0.12 --fp16=False
#   になります。
# - 本スクリプトは上記を保ち、1 GPUでメモリに収めるために
#     --batch-gpu=128
#   を使います。これは有効バッチサイズ --batch=512 を変えず、
#   勾配蓄積の分割サイズだけを変えるため、論文設定からのズレは小さいです。
# - 評価スケジュールは 3k, 10k, 20k, 40k, 80k, 120k, 160k, 200k。
#   途中は1 seed範囲で軽く評価し、最終200kだけ追加で3 seed正式評価します。
# - 追加している途中FID評価、health記録、sample生成は、保存済みsnapshotを
#   評価するだけで学習重みを変更しません。
# - 指定kimgごとにプロセスを切る点は、完全な連続実行とは少し違います。
#   そのズレを小さくするため、同じ seed を各再開で使います。

cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
RUN_DIR="${RUN_DIR:-training-runs/cifar10-cond-ddpmpp-edm-managed}"
FID_ROOT="${FID_ROOT:-fid-evals/cifar10-cond-ddpmpp-edm-managed}"
LOG_DIR="${LOG_DIR:-logs}"

TOTAL_KIMG="${TOTAL_KIMG:-200000}"
EVAL_KIMGS="${EVAL_KIMGS:-3000 10000 20000 40000 80000 120000 160000 200000}"
TICK_KIMG="${TICK_KIMG:-50}"
SNAP_TICKS="${SNAP_TICKS:-50}"
DUMP_TICKS="${DUMP_TICKS:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
SEED="${SEED:-}"
FID_SEEDS="${FID_SEEDS:-0-49999}"
FINAL_FID_SEEDS_LIST="${FINAL_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
FID_REF="${FID_REF:-https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-1}"

MASTER_LOG="$LOG_DIR/cifar10-cond-train-eval-resume.log"
SUMMARY="$FID_ROOT/summary.tsv"
HEALTH="$FID_ROOT/health.tsv"
STOP_FILE="$RUN_DIR/STOP_AFTER_CURRENT_STAGE"

mkdir -p "$RUN_DIR" "$FID_ROOT" "$LOG_DIR"

if [[ -z "$SEED" && -f "$RUN_DIR/training_options.json" ]]; then
    SEED="$("$PYTHON" - "$RUN_DIR/training_options.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("seed", ""))
PY
)"
fi
SEED="${SEED:-0}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

kimg_to_mimg() {
    awk -v kimg="$1" 'BEGIN { printf "%.6g", kimg / 1000 }'
}

state_kimg() {
    basename "$1" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/' | sed 's/^0*//; s/^$/0/'
}

latest_state() {
    find "$RUN_DIR" -maxdepth 1 -type f -name 'training-state-*.pt' 2>/dev/null | sort | tail -n 1
}

snapshot_path() {
    printf '%s/network-snapshot-%06d.pkl' "$RUN_DIR" "$1"
}

state_path() {
    printf '%s/training-state-%06d.pt' "$RUN_DIR" "$1"
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'kimg\tseeds\tfid\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'kimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

fid_already_done() {
    local target="$1"
    local seeds="${2:-$FID_SEEDS}"
    ensure_summary
    awk -F '\t' -v target="$target" -v seeds="$seeds" \
        'NR > 1 && $1 == target && $2 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

health_already_done() {
    local target="$1"
    ensure_health
    awk -F '\t' -v target="$target" \
        'NR > 1 && $1 == target { found = 1 } END { exit(found ? 0 : 1) }' \
        "$HEALTH"
}

run_train_to() {
    local target="$1"
    local duration
    duration="$(kimg_to_mimg "$target")"

    local resume_args=()
    local state
    state="$(latest_state || true)"
    if [[ -n "$state" ]]; then
        resume_args+=(--resume="$state")
        log "Resuming from $state toward ${target} kimg"
    else
        log "Starting fresh toward ${target} kimg"
    fi

    "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        --outdir="$RUN_DIR" --nosubdir \
        --data="$DATASET" \
        --cond=1 --arch=ddpmpp \
        --batch=512 --batch-gpu="$BATCH_GPU" \
        --seed="$SEED" \
        --tick="$TICK_KIMG" --snap="$SNAP_TICKS" --dump="$DUMP_TICKS" \
        --duration="$duration" \
        "${resume_args[@]}" 2>&1 | tee -a "$MASTER_LOG"
}

record_health_for() {
    local target="$1"
    local state
    local snapshot
    local log_line
    local sec_per_kimg
    local gpumem
    local reserved
    local latest_loss
    local loss_status

    state="$(state_path "$target")"
    snapshot="$(snapshot_path "$target")"

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

path = sys.argv[1]
value = ""
try:
    with open(path, "r", encoding="utf-8") as f:
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
        "${gpumem:-NA}" "${reserved:-NA}" "$state" "$snapshot" >> "$HEALTH"

    log "Recorded health for ${target} kimg: loss=${latest_loss:-NA} status=$loss_status sec/kimg=${sec_per_kimg:-NA} gpumem=${gpumem:-NA} reserved=${reserved:-NA}"
}

generate_samples_for() {
    local target="$1"
    local snapshot
    local sample_dir

    if [[ "$GENERATE_SAMPLES" != "1" ]]; then
        log "Sample generation disabled for ${target} kimg"
        return 0
    fi

    snapshot="$(snapshot_path "$target")"
    sample_dir="$FID_ROOT/samples-${target}-${SAMPLE_SEEDS}"

    if [[ -d "$sample_dir" ]]; then
        log "Samples already exist for ${target} kimg seeds ${SAMPLE_SEEDS}; skipping"
        return 0
    fi

    if [[ ! -f "$snapshot" ]]; then
        log "Missing snapshot $snapshot; cannot generate samples"
        return 1
    fi

    log "Generating sample images for ${target} kimg seeds ${SAMPLE_SEEDS}"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$sample_dir" \
        --seeds="$SAMPLE_SEEDS" \
        --steps=18 \
        --network="$snapshot" 2>&1 | tee -a "$MASTER_LOG"
}

run_fid_for() {
    local target="$1"
    local seeds="${2:-$FID_SEEDS}"
    local snapshot
    local images
    local fid_log
    local fid

    snapshot="$(snapshot_path "$target")"
    images="$FID_ROOT/images-${target}-${seeds}"
    fid_log="$FID_ROOT/fid-${target}-${seeds}.log"

    if [[ ! -f "$snapshot" ]]; then
        log "Missing snapshot $snapshot; cannot evaluate FID"
        return 1
    fi

    if fid_already_done "$target" "$seeds"; then
        log "FID already recorded for ${target} kimg seeds ${seeds}; skipping"
        return 0
    fi

    log "Generating FID images for ${target} kimg seeds ${seeds}"
    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --outdir="$images" \
        --seeds="$seeds" \
        --subdirs \
        --steps=18 \
        --network="$snapshot" 2>&1 | tee -a "$fid_log"

    log "Calculating FID for ${target} kimg seeds ${seeds}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" 2>&1 | tee -a "$fid_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "Could not parse FID from $fid_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\n' "$target" "$seeds" "$fid" "$snapshot" "$images" >> "$SUMMARY"
    log "Recorded FID ${fid} for ${target} kimg seeds ${seeds}"
}

main() {
    log "Using CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    log "Run dir: $RUN_DIR"
    log "FID summary: $SUMMARY"
    log "Batch GPU: $BATCH_GPU; seed: $SEED; eval kimgs: $EVAL_KIMGS; total: $TOTAL_KIMG kimg"
    log "To request a clean pause after the current stage, create: $STOP_FILE"

    local target
    for target in $EVAL_KIMGS; do
        if (( target > TOTAL_KIMG )); then
            log "Skipping ${target} kimg because it is beyond TOTAL_KIMG=${TOTAL_KIMG}"
            continue
        fi

        local state
        local current=0
        state="$(latest_state || true)"
        if [[ -n "$state" ]]; then
            current="$(state_kimg "$state")"
        fi

        if (( current < target )); then
            run_train_to "$target"
        else
            log "Already have training state at ${current} kimg; skipping train to ${target} kimg"
        fi

        if [[ ! -f "$(state_path "$target")" ]]; then
            log "Expected state $(state_path "$target") was not found"
            exit 1
        fi

        record_health_for "$target"
        generate_samples_for "$target"
        run_fid_for "$target" "$FID_SEEDS"

        if (( target == TOTAL_KIMG )); then
            local final_seeds
            for final_seeds in $FINAL_FID_SEEDS_LIST; do
                run_fid_for "$target" "$final_seeds"
            done
        fi

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed stage ${target} kimg"
            exit 0
        fi
    done

    log "All stages complete"
    log "Health:"
    ensure_health
    cat "$HEALTH" | tee -a "$MASTER_LOG"
    log "Summary:"
    ensure_summary
    cat "$SUMMARY" | tee -a "$MASTER_LOG"
}

main "$@"
