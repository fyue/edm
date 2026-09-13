#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 7-1:
# CIFAR-10 class-conditional の linear-path flow matching (FM) baseline を、
# EDM実験と同じDDPM++構造・学習設定で段階的に学習・評価します。
#
# 経路と目標速度:
#   x_t = (1 - t) z + t y,  v_target = y - z,  t ~ U[0,1),  z ~ N(0,I)
# サンプリングは t=0 (noise) -> t=1 (data) の向きに積分します。
#
# 段階:
#   PROFILE=smoke  50 kimg まで学習し、sample生成・決定性確認・checkpoint再読込
#                  までを行う短時間の動作確認。
#   PROFILE=pilot  1k/3k/10k/20k kimg でsampleとFID-5k。
#   PROFILE=full   40k/80k/120k/160k/200k kimg へresumeし、最終kimgで
#                  FID-50k×3 seed範囲まで実行。
#
# smoke以外は明示的に PROFILE を指定しない限り起動しません。
#
# 使い方 (GPU待機付き):
#   PROFILE=smoke experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/schedule_after_gpu_free.sh
#
# 計画表示のみ (GPU不要):
#   PLAN_ONLY=1 PROFILE=full \
#     experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/run_fm_linear_baseline.sh
#
# 現在のstageが終わったら安全に止めたい場合:
#   touch experiments/7/1_cifar10_cond_linear_fm_baseline/STOP_AFTER_CURRENT_STAGE
#
# 再実行すると、最新の training-state-*.pt から再開し、記録済みの
# health・sample・FIDはスキップします。

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

EXP_ROOT="${EXP_ROOT:-experiments/7/1_cifar10_cond_linear_fm_baseline}"
CONFIG="${CONFIG:-$EXP_ROOT/config/profiles.tsv}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-fm-linear}"

PROFILE="${PROFILE:-smoke}"
PLAN_ONLY="${PLAN_ONLY:-0}"

# 学習設定。EDM実験 (experiments/1, experiments/3) と揃えています。
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

# 生成・評価設定。
SAMPLER="${SAMPLER:-fm-euler}"
STEPS="${STEPS:-18}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-1}"

MASTER_LOG="$LOG_DIR/fm-linear-baseline-$PROFILE.log"
SUMMARY="$RESULTS_DIR/fm_linear_baseline_fid_summary.tsv"
HEALTH="$RESULTS_DIR/fm_linear_baseline_health.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_STAGE"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

is_url() {
    [[ "$1" =~ ^https?:// ]]
}

#----------------------------------------------------------------------------
# Profile settings.

load_profile() {
    local line
    line="$(awk -F '\t' -v profile="$PROFILE" \
        '$1 == profile { print; exit }' "$CONFIG")"
    if [[ -z "$line" ]]; then
        printf 'Unknown PROFILE=%s; available profiles in %s:\n' "$PROFILE" "$CONFIG" >&2
        awk -F '\t' '$1 !~ /^#/ && NF > 1 { print "  " $1 }' "$CONFIG" >&2
        exit 1
    fi
    IFS=$'\t' read -r _ STAGE_KIMGS TICK_KIMG SNAP_TICKS DUMP_TICKS \
        FID_NUM FID_SEEDS FINAL_FID_NUM FINAL_FID_SEEDS_LIST VERIFY <<< "$line"
}

#----------------------------------------------------------------------------
# Result tables.

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'profile\tkimg\tsampler\tsteps\tnfe\tnum_images\tseeds\tfid\tsnapshot\timages\n' > "$SUMMARY"
    fi
}

ensure_health() {
    if [[ ! -f "$HEALTH" ]]; then
        printf 'profile\tkimg\tlatest_loss\tloss_status\tsec_per_kimg\tgpumem_gb\treserved_gb\tstate\tsnapshot\n' > "$HEALTH"
    fi
}

fid_already_done() {
    local kimg="$1"
    local seeds="$2"
    local num="$3"
    ensure_summary
    awk -F '\t' -v kimg="$kimg" -v seeds="$seeds" -v num="$num" \
        -v sampler="$SAMPLER" -v steps="$STEPS" \
        'NR > 1 && $2 == kimg && $3 == sampler && $4 == steps && $6 == num && $7 == seeds { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

health_already_done() {
    local kimg="$1"
    ensure_health
    awk -F '\t' -v kimg="$kimg" \
        'NR > 1 && $2 == kimg { found = 1 } END { exit(found ? 0 : 1) }' \
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

run_train_to() {
    local target="$1"
    local duration
    duration="$(kimg_to_mimg "$target")"

    local resume_args=()
    local state
    state="$(latest_state || true)"
    if [[ -n "$state" ]]; then
        resume_args+=(--resume="$state")
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] train --precond=$PRECOND --duration=$duration ${resume_args[*]:-(fresh start)}"
        return 0
    fi

    if [[ -n "$state" ]]; then
        log "Resuming from $state toward ${target} kimg"
    else
        log "Starting fresh toward ${target} kimg"
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$PROFILE" "$target" "${latest_loss:-NA}" "$loss_status" "${sec_per_kimg:-NA}" \
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
    local snapshot
    local sample_dir

    snapshot="$(snapshot_path "$target")"
    sample_dir="$IMAGE_ROOT/samples-${target}-${SAMPLE_SEEDS}"

    if [[ "$GENERATE_SAMPLES" != "1" ]]; then
        log "Sample generation disabled for ${target} kimg"
        return 0
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] generate --sampler=$SAMPLER --steps=$STEPS --seeds=$SAMPLE_SEEDS -> $sample_dir"
        return 0
    fi
    if [[ -d "$sample_dir" ]]; then
        log "Samples already exist for ${target} kimg seeds ${SAMPLE_SEEDS}; skipping"
        return 0
    fi
    if [[ ! -f "$snapshot" ]]; then
        log "Missing snapshot $snapshot; cannot generate samples"
        return 1
    fi

    log "Generating samples for ${target} kimg seeds ${SAMPLE_SEEDS} ($SAMPLER, ${STEPS} steps, NFE=$(nfe_for "$STEPS"))"
    mkdir -p "$sample_dir"
    generate_images "$snapshot" "$sample_dir" "$SAMPLE_SEEDS" \
        "$LOG_DIR/samples-${target}-${SAMPLE_SEEDS}.log" 0
}

verify_sample_determinism() {
    # 同じseedで生成し直した画像が1 byteまで一致することを確認する。
    local target="$1"
    local snapshot
    local sample_dir
    local repeat_dir
    local run_log

    snapshot="$(snapshot_path "$target")"
    sample_dir="$IMAGE_ROOT/samples-${target}-${SAMPLE_SEEDS}"
    repeat_dir="$IMAGE_ROOT/samples-${target}-${SAMPLE_SEEDS}-repeat"
    run_log="$LOG_DIR/samples-${target}-${SAMPLE_SEEDS}-repeat.log"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] regenerate seeds ${SAMPLE_SEEDS} and compare checksums"
        return 0
    fi
    if [[ ! -d "$sample_dir" ]]; then
        log "Missing $sample_dir; cannot verify determinism"
        return 1
    fi

    log "Regenerating seeds ${SAMPLE_SEEDS} to verify determinism"
    rm -rf "$repeat_dir"
    mkdir -p "$repeat_dir"
    generate_images "$snapshot" "$repeat_dir" "$SAMPLE_SEEDS" "$run_log" 0

    local first_count
    local second_count
    first_count="$(find "$sample_dir" -type f -name '*.png' | wc -l | tr -d ' ')"
    second_count="$(find "$repeat_dir" -type f -name '*.png' | wc -l | tr -d ' ')"
    if [[ "$first_count" != "$second_count" ]]; then
        log "Determinism check failed: ${first_count} vs ${second_count} images"
        return 1
    fi

    if diff <(cd "$sample_dir" && find . -type f -name '*.png' | sort | xargs sha256sum) \
            <(cd "$repeat_dir" && find . -type f -name '*.png' | sort | xargs sha256sum) > /dev/null; then
        log "Determinism check passed: ${first_count} images match byte-for-byte"
        rm -rf "$repeat_dir"
        return 0
    fi

    log "Determinism check failed: regenerated images differ (kept $repeat_dir)"
    return 1
}

verify_checkpoint_for() {
    local target="$1"

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [${target} kimg] verify checkpoint reload, loss/gradient finiteness"
        return 0
    fi

    log "Verifying checkpoint reload and training health at ${target} kimg"
    "$PYTHON" "$EXP_ROOT/scripts/verify_fm_checkpoint.py" \
        --run-dir="$RUN_DIR" --kimg="$target" 2>&1 | tee -a "$MASTER_LOG"

    log "Verifying that --resume resolves for ${target} kimg"
    "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        --outdir="$RUN_DIR" --nosubdir \
        --data="$DATASET" \
        --cond=1 --arch="$ARCH" --precond="$PRECOND" \
        --batch="$BATCH" --batch-gpu="$BATCH_GPU" \
        --seed="$SEED" \
        --resume="$(state_path "$target")" \
        --dry-run 2>&1 | tail -n 12 | tee -a "$MASTER_LOG"
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
    images="$IMAGE_ROOT/fid-${target}-n${num}-seeds-${seeds}"
    run_log="$LOG_DIR/fid-${target}-n${num}-seeds-${seeds}.log"

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

    log "Calculating FID for ${target} kimg seeds ${seeds}"
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$PROFILE" "$target" "$SAMPLER" "$STEPS" "$(nfe_for "$STEPS")" \
        "$num" "$seeds" "$fid" "$snapshot" "$images" >> "$SUMMARY"
    log "Recorded FID ${fid} for ${target} kimg seeds ${seeds}"
}

#----------------------------------------------------------------------------

check_inputs() {
    local missing=0
    if [[ ! -f "$CONFIG" ]]; then
        printf 'Missing required file: %s\n' "$CONFIG" >&2
        missing=1
    fi
    if [[ ! -f "$DATASET" ]]; then
        printf 'Missing required file: %s\n' "$DATASET" >&2
        missing=1
    fi
    if [[ "$FID_NUM" != "0" || "$FINAL_FID_NUM" != "0" ]]; then
        if [[ ! -f "$FID_REF" ]] && ! is_url "$FID_REF"; then
            printf 'Missing required file: %s\n' "$FID_REF" >&2
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
    load_profile
    check_inputs

    local last_stage
    last_stage="$(awk '{ print $NF }' <<< "$STAGE_KIMGS")"

    log "Profile: $PROFILE"
    log "Stages: $STAGE_KIMGS kimg (tick=$TICK_KIMG snap=$SNAP_TICKS dump=$DUMP_TICKS)"
    log "Run dir: $RUN_DIR"
    log "Sampler: $SAMPLER steps=$STEPS NFE=$(nfe_for "$STEPS") sample seeds=$SAMPLE_SEEDS"
    log "Training: arch=$ARCH precond=$PRECOND batch=$BATCH batch-gpu=$BATCH_GPU lr=$LR ema=$EMA dropout=$DROPOUT augment=$AUGMENT fp16=$FP16 seed=$SEED"
    log "Stage FID: num=$FID_NUM seeds=$FID_SEEDS; final FID: num=$FINAL_FID_NUM seeds=$FINAL_FID_SEEDS_LIST"
    log "Results: $SUMMARY / $HEALTH"
    log "To pause after the current stage, create: $STOP_FILE"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no GPU work will be started"
    fi

    local target
    for target in $STAGE_KIMGS; do
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

        if [[ "$PLAN_ONLY" != "1" && ! -f "$(state_path "$target")" ]]; then
            log "Expected state $(state_path "$target") was not found"
            exit 1
        fi

        record_health_for "$target"
        generate_samples_for "$target"
        if [[ "$VERIFY" == "1" ]]; then
            verify_sample_determinism "$target"
            verify_checkpoint_for "$target"
        fi
        run_fid_for "$target" "$FID_SEEDS" "$FID_NUM"

        if [[ "$target" == "$last_stage" && "$FINAL_FID_NUM" != "0" ]]; then
            local final_seeds
            for final_seeds in $FINAL_FID_SEEDS_LIST; do
                run_fid_for "$target" "$final_seeds" "$FINAL_FID_NUM"
            done
        fi

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after completed stage ${target} kimg"
            exit 0
        fi
    done

    log "Profile $PROFILE complete. Later profiles are not started automatically."
    if [[ "$PLAN_ONLY" != "1" ]]; then
        log "Health:"
        ensure_health
        cat "$HEALTH" | tee -a "$MASTER_LOG"
        if [[ "$FID_NUM" != "0" || "$FINAL_FID_NUM" != "0" ]]; then
            log "FID summary:"
            ensure_summary
            cat "$SUMMARY" | tee -a "$MASTER_LOG"
        fi
    fi
}

main "$@"
