#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 7-2:
# linear-path flow matching (FM) baseline の pilot 学習・評価を行います。
#
# Experiment 7-1 (実装 + 50 kimg smoke) の checkpoint を起点に、
# 1k / 3k / 10k / 20k kimg まで学習を進め、各stageで
# sample 64枚 と FID-5k を記録します。
#
# 経路と目標速度 (7-1と同一):
#   x_t = (1 - t) z + t y,  v_target = y - z,  t ~ U[0,1),  z ~ N(0,I)
# サンプリングは t=0 (noise) -> t=1 (data) の向きに積分します。
#
# 学習設定は7-1のsmokeと同一で、重みとoptimizer状態を引き継ぎます。
# EDMのtraining-stateは乱数状態を保存しないため、未中断runとbit単位で同じ
# 学習軌道になることまでは保証しません。
# (arch=ddpmpp, batch=512, batch-gpu=128, lr=0.001, ema=0.5,
#  dropout=0.13, augment=0.12, fp32, seed=0)
#
# 使い方 (GPU待機付き、tmux推奨):
#   tmux new -s edm-exp7-2
#   experiments/7/2_cifar10_cond_linear_fm_pilot/scripts/schedule_after_gpu_free.sh
#
# 計画表示のみ (GPU不要):
#   PLAN_ONLY=1 experiments/7/2_cifar10_cond_linear_fm_pilot/scripts/run_fm_linear_pilot.sh
#
# 進捗確認:
#   tail -f experiments/7/2_cifar10_cond_linear_fm_pilot/logs/fm-linear-pilot.log
#
# 結果確認:
#   cat experiments/7/2_cifar10_cond_linear_fm_pilot/results/fm_linear_pilot_fid_summary.tsv
#   cat experiments/7/2_cifar10_cond_linear_fm_pilot/results/fm_linear_pilot_health.tsv
#
# 現在のstageが終わったら安全に止めたい場合:
#   touch experiments/7/2_cifar10_cond_linear_fm_pilot/STOP_AFTER_CURRENT_STAGE
#
# 再実行すると、このstage runの最新 training-state-*.pt から再開し、
# 記録済みのhealth・sample・FIDはスキップします。
#
# 完全に新規初期化から始めたい場合は BASE_STATE='' を指定します。

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

EXP_ROOT="${EXP_ROOT:-experiments/7/2_cifar10_cond_linear_fm_pilot}"
CONFIG="${CONFIG:-$EXP_ROOT/config/stages.tsv}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-fm-linear-pilot}"

# 起点となる Experiment 7-1 の smoke checkpoint。
# BASE_STATE='' にすると新規初期化から学習する。
BASE_RUN="${BASE_RUN:-experiments/7/1_cifar10_cond_linear_fm_baseline/training-runs/cifar10-cond-ddpmpp-fm-linear}"
BASE_STATE="${BASE_STATE-$BASE_RUN/training-state-000050.pt}"

PLAN_ONLY="${PLAN_ONLY:-0}"

# 学習設定。7-1のsmokeと同一に保つこと (同じ学習軌道の続きにするため)。
DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
ARCH="${ARCH:-ddpmpp}"
PRECOND="${PRECOND:-fm}"
BATCH="${BATCH:-512}"
BATCH_GPU="${BATCH_GPU:-128}"
LR="${LR:-0.001}"
EMA="${EMA:-0.5}"
DROPOUT="${DROPOUT:-0.13}"
AUGMENT="${AUGMENT:-0.12}"
FP16="${FP16:-0}"
SEED="${SEED:-0}"
TICK_KIMG="${TICK_KIMG:-50}"
SNAP_TICKS="${SNAP_TICKS:-50}"
DUMP_TICKS="${DUMP_TICKS:-50}"

# 生成・評価設定。
SAMPLER="${SAMPLER:-fm-euler}"
STEPS="${STEPS:-18}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"

MASTER_LOG="$LOG_DIR/fm-linear-pilot.log"
SUMMARY="$RESULTS_DIR/fm_linear_pilot_fid_summary.tsv"
HEALTH="$RESULTS_DIR/fm_linear_pilot_health.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_STAGE"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

is_url() {
    [[ "$1" =~ ^https?:// ]]
}

#----------------------------------------------------------------------------
# Result tables.

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'kimg\tsampler\tsteps\tnfe\tnum_images\tseeds\tfid\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'kimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

fid_already_done() {
    local kimg="$1"
    local seeds="$2"
    local num="$3"
    ensure_summary
    awk -F '\t' -v kimg="$kimg" -v seeds="$seeds" -v num="$num" \
        -v sampler="$SAMPLER" -v steps="$STEPS" \
        'NR > 1 && $1 == kimg && $2 == sampler && $3 == steps && $5 == num && $6 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

health_already_done() {
    local kimg="$1"
    ensure_health
    awk -F '\t' -v kimg="$kimg" \
        'NR > 1 && $1 == kimg { found = 1 } END { exit(found ? 0 : 1) }' \
        "$HEALTH"
}

#----------------------------------------------------------------------------
# Training helpers.

nfe_for() {
    # Euler は N NFE、Heun は最後の区間だけEulerなので 2N-1 NFE。
    local steps="$1"
    case "$SAMPLER" in
        fm-euler) printf '%s' "$steps" ;;
        fm-heun)  printf '%s' "$(( 2 * steps - 1 ))" ;;
        *)        printf 'NA' ;;
    esac
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

# このstage runの最新state、無ければ7-1の起点stateを返す。
resume_state_for() {
    local state
    state="$(latest_state || true)"
    if [[ -n "$state" ]]; then
        printf '%s' "$state"
        return 0
    fi
    if [[ -n "$BASE_STATE" ]]; then
        printf '%s' "$BASE_STATE"
    fi
}

current_kimg() {
    local state
    state="$(resume_state_for)"
    if [[ -z "$state" ]]; then
        printf '0'
        return 0
    fi
    state_kimg "$state"
}

run_train_to() {
    local target="$1"
    local planned_start="${2:-}"
    local planned_resume="${3:-}"
    local duration
    local resume_state
    local start_kimg

    duration="$(kimg_to_mimg "$target")"
    resume_state="$(resume_state_for)"
    start_kimg="$(current_kimg)"

    local resume_args=()
    if [[ -n "$resume_state" ]]; then
        resume_args+=(--resume="$resume_state")
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        if [[ -n "$planned_start" ]]; then
            start_kimg="$planned_start"
            resume_state="$planned_resume"
        fi
        if [[ -n "$resume_state" ]]; then
            log "PLAN [${target} kimg] train ${start_kimg} -> ${target} kimg, resume=$resume_state"
        else
            log "PLAN [${target} kimg] train 0 -> ${target} kimg, fresh initialization"
        fi
        return 0
    fi

    if [[ -n "$resume_state" ]]; then
        log "Training ${start_kimg} -> ${target} kimg (resume=$resume_state)"
    else
        log "Training 0 -> ${target} kimg (fresh initialization)"
    fi

    mkdir -p "$RUN_DIR"
    "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        --outdir="$RUN_DIR" --nosubdir \
        --data="$DATASET" \
        --cond=1 --arch="$ARCH" --precond="$PRECOND" \
        --batch="$BATCH" --batch-gpu="$BATCH_GPU" \
        --lr="$LR" --ema="$EMA" --dropout="$DROPOUT" --augment="$AUGMENT" \
        --fp16="$FP16" \
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

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] record health -> $HEALTH"
        return 0
    fi

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

    if [[ "$loss_status" != "ok" ]]; then
        log "Loss is not finite at ${target} kimg; stopping"
        return 1
    fi
}

#----------------------------------------------------------------------------
# Sampling and evaluation.

generate_images() {
    local snapshot="$1"
    local outdir="$2"
    local seeds="$3"
    local run_log="$4"
    local subdirs="$5"

    local subdir_args=()
    if [[ "$subdirs" == "1" ]]; then
        subdir_args+=(--subdirs)
    fi

    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --network="$snapshot" \
        --outdir="$outdir" \
        --seeds="$seeds" \
        --batch="$GEN_BATCH" \
        --sampler="$SAMPLER" \
        --steps="$STEPS" \
        "${subdir_args[@]}" 2>&1 | tee -a "$run_log"
}

generate_samples_for() {
    local target="$1"
    local seeds="$2"
    local snapshot
    local sample_dir
    local complete_marker

    if [[ "$seeds" == "-" ]]; then
        return 0
    fi

    snapshot="$(snapshot_path "$target")"
    sample_dir="$IMAGE_ROOT/samples-${target}-${SAMPLER}-steps${STEPS}-seeds-${seeds}"
    complete_marker="$sample_dir/.complete"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] generate --sampler=$SAMPLER --steps=$STEPS --seeds=$seeds -> $sample_dir"
        return 0
    fi
    if [[ -f "$complete_marker" ]]; then
        log "Samples already exist for ${target} kimg seeds ${seeds}; skipping"
        return 0
    fi
    if [[ ! -f "$snapshot" ]]; then
        log "Missing snapshot $snapshot; cannot generate samples"
        return 1
    fi

    log "Generating samples for ${target} kimg seeds ${seeds} ($SAMPLER, ${STEPS} steps, NFE=$(nfe_for "$STEPS"))"
    mkdir -p "$sample_dir"
    generate_images "$snapshot" "$sample_dir" "$seeds" \
        "$LOG_DIR/samples-${target}-${seeds}.log" 0
    touch "$complete_marker"
}

run_fid_for() {
    local target="$1"
    local seeds="$2"
    local num="$3"
    local snapshot
    local images
    local run_log
    local fid

    if [[ "$num" == "0" || "$seeds" == "-" ]]; then
        return 0
    fi

    snapshot="$(snapshot_path "$target")"
    images="$IMAGE_ROOT/fid-${target}-${SAMPLER}-steps${STEPS}-n${num}-seeds-${seeds}"
    run_log="$LOG_DIR/fid-${target}-${SAMPLER}-steps${STEPS}-n${num}-seeds-${seeds}.log"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] generate ${num} images (seeds ${seeds}) -> $images"
        log "PLAN [${target} kimg] fid.py calc --num=${num} --ref=$FID_REF"
        return 0
    fi
    if [[ ! -f "$snapshot" ]]; then
        log "Missing snapshot $snapshot; cannot evaluate FID"
        return 1
    fi
    if fid_already_done "$target" "$seeds" "$num"; then
        log "FID already recorded for ${target} kimg seeds ${seeds}; skipping"
        return 0
    fi

    local png_count=0
    mkdir -p "$images"
    png_count="$(find "$images" -type f -name '*.png' | wc -l | tr -d ' ')"
    if (( png_count >= num )); then
        log "Found ${png_count} pngs in $images; skipping generation"
    else
        log "Generating ${num} images for ${target} kimg seeds ${seeds}"
        generate_images "$snapshot" "$images" "$seeds" "$run_log" 1
    fi

    log "Calculating FID-${num} for ${target} kimg seeds ${seeds}"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$num" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "Could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$target" "$SAMPLER" "$STEPS" "$(nfe_for "$STEPS")" \
        "$num" "$seeds" "$fid" "$snapshot" "$images" >> "$SUMMARY"
    log "Recorded FID-${num} ${fid} for ${target} kimg seeds ${seeds}"
}

#----------------------------------------------------------------------------

check_inputs() {
    local missing=0
    if [[ "$PRECOND" != "fm" ]]; then
        printf 'PRECOND must remain fm for Experiment 7-2: %s\n' "$PRECOND" >&2
        missing=1
    fi
    if [[ "$SAMPLER" != "fm-euler" && "$SAMPLER" != "fm-heun" ]]; then
        printf 'SAMPLER must be fm-euler or fm-heun: %s\n' "$SAMPLER" >&2
        missing=1
    fi
    if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
        printf 'STEPS must be a positive integer: %s\n' "$STEPS" >&2
        missing=1
    fi
    if [[ ! -f "$CONFIG" ]]; then
        printf 'Missing required file: %s\n' "$CONFIG" >&2
        missing=1
    fi
    if [[ ! -f "$DATASET" ]]; then
        printf 'Missing required file: %s\n' "$DATASET" >&2
        missing=1
    fi
    if [[ ! -f "$FID_REF" ]] && ! is_url "$FID_REF"; then
        printf 'Missing required file: %s\n' "$FID_REF" >&2
        missing=1
    fi
    # 起点stateは、このstage runがまだ何も持っていないときだけ必要。
    if [[ -n "$BASE_STATE" && -z "$(latest_state || true)" ]]; then
        if [[ ! -f "$BASE_STATE" ]]; then
            printf 'Missing base checkpoint: %s\n' "$BASE_STATE" >&2
            printf '  (set BASE_STATE="" to train from scratch instead)\n' >&2
            missing=1
        fi
        local base_snapshot
        base_snapshot="$(dirname "$BASE_STATE")/network-snapshot-$(basename "$BASE_STATE" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/').pkl"
        if [[ ! -f "$base_snapshot" ]]; then
            printf 'Missing base snapshot next to BASE_STATE: %s\n' "$base_snapshot" >&2
            missing=1
        fi
    fi
    if [[ "$PLAN_ONLY" != "1" ]]; then
        if ! command -v "$TORCHRUN" >/dev/null 2>&1; then
            printf 'Missing executable torchrun: %s\n' "$TORCHRUN" >&2
            missing=1
        fi
        if ! command -v "$PYTHON" >/dev/null 2>&1; then
            printf 'Missing executable python: %s\n' "$PYTHON" >&2
            missing=1
        fi
    fi
    if (( missing != 0 )); then
        exit 1
    fi
}

main() {
    check_inputs

    log "Experiment 7-2: linear FM pilot"
    log "Run dir: $RUN_DIR"
    if [[ -n "$BASE_STATE" ]]; then
        log "Base checkpoint: $BASE_STATE"
    else
        log "Base checkpoint: (none; training from scratch)"
    fi
    log "Config: $CONFIG"
    log "Training: arch=$ARCH precond=$PRECOND batch=$BATCH batch-gpu=$BATCH_GPU lr=$LR ema=$EMA dropout=$DROPOUT augment=$AUGMENT fp16=$FP16 seed=$SEED"
    log "Ticks: tick=$TICK_KIMG snap=$SNAP_TICKS dump=$DUMP_TICKS"
    log "Sampler: $SAMPLER steps=$STEPS NFE=$(nfe_for "$STEPS")"
    log "Results: $SUMMARY / $HEALTH"
    log "To pause after the current stage, create: $STOP_FILE"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no GPU work will be started"
    fi

    local kimg
    local fid_num
    local fid_seeds
    local sample_seeds
    local planned_have
    local planned_resume

    planned_have="$(current_kimg)"
    planned_resume="$(resume_state_for)"

    while IFS=$'\t' read -r kimg fid_num fid_seeds sample_seeds; do
        [[ -z "${kimg:-}" || "$kimg" =~ ^# ]] && continue

        local have=0
        if [[ "$PLAN_ONLY" == "1" ]]; then
            have="$planned_have"
        else
            have="$(current_kimg)"
        fi
        if (( have < kimg )); then
            if [[ "$PLAN_ONLY" == "1" ]]; then
                run_train_to "$kimg" "$planned_have" "$planned_resume"
                planned_have="$kimg"
                planned_resume="$(state_path "$kimg")"
            else
                run_train_to "$kimg"
            fi
        else
            log "Already at ${have} kimg; skipping train to ${kimg} kimg"
        fi

        if [[ "$PLAN_ONLY" != "1" && ! -f "$(state_path "$kimg")" ]]; then
            log "Expected state $(state_path "$kimg") was not found"
            exit 1
        fi

        record_health_for "$kimg"
        generate_samples_for "$kimg" "$sample_seeds"
        run_fid_for "$kimg" "$fid_seeds" "$fid_num"

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed stage ${kimg} kimg"
            exit 0
        fi
    done < "$CONFIG"

    log "Pilot complete. Full (40k-200k) は自動起動しません。"
    if [[ "$PLAN_ONLY" != "1" ]]; then
        log "Health:"
        ensure_health
        cat "$HEALTH" | tee -a "$MASTER_LOG"
        log "FID summary:"
        ensure_summary
        cat "$SUMMARY" | tee -a "$MASTER_LOG"
    fi
}

main "$@"
