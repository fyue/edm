#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 0-6: train on the 0-2 subset of 30k images in one process, as in
# 0-1. No FID evaluation runs during training; use run_fid.sh afterwards.

cd "$(dirname "$0")/../../../.."

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/0/6_cifar10_cond_edm_30k_subset_continuous}"
CHECKPOINT_CONFIG="${CHECKPOINT_CONFIG:-$EXP_ROOT/config/checkpoint_kimgs.txt}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
DATASET_MAX_SIZE=30000
DATASET_RANDOM_SEED=0
TARGET_KIMG=100000
TICK_KIMG="${TICK_KIMG:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
SEED=0
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/train.log"
MANIFEST="$RESULTS_DIR/checkpoint_manifest.tsv"

log() {
    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf '[PLAN] %s\n' "$*"
    else
        printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
    fi
}

fail() {
    printf '%s\n' "$*" >&2
    exit 1
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

load_checkpoints() {
    [[ -f "$CHECKPOINT_CONFIG" ]] || fail "Missing checkpoint config: $CHECKPOINT_CONFIG"
    mapfile -t CHECKPOINT_KIMGS < <(awk '!/^#/ && NF { print $1 }' "$CHECKPOINT_CONFIG")
    (( ${#CHECKPOINT_KIMGS[@]} > 0 )) || fail "No checkpoints in $CHECKPOINT_CONFIG"
    local index target
    for index in "${!CHECKPOINT_KIMGS[@]}"; do
        target="${CHECKPOINT_KIMGS[$index]}"
        (( target == (index + 1) * 10000 )) || fail "Unexpected checkpoint at line $((index + 1)): $target"
    done
    (( CHECKPOINT_KIMGS[${#CHECKPOINT_KIMGS[@]} - 1] == TARGET_KIMG )) || \
        fail "Final checkpoint must equal TARGET_KIMG=$TARGET_KIMG"
}

verify_outputs() {
    local target
    for target in "${CHECKPOINT_KIMGS[@]}"; do
        [[ -s "$(snapshot_path "$target")" ]] || fail "Missing snapshot: $(snapshot_path "$target")"
        [[ -s "$(state_path "$target")" ]] || fail "Missing state: $(state_path "$target")"
    done
}

write_manifest() {
    local target snapshot state
    printf 'kimg\tsnapshot\tsnapshot_bytes\tstate\tstate_bytes\n' > "$MANIFEST"
    for target in "${CHECKPOINT_KIMGS[@]}"; do
        snapshot="$(snapshot_path "$target")"
        state="$(state_path "$target")"
        printf '%d\t%s\t%s\t%s\t%s\n' \
            "$target" "$snapshot" "$(stat -c '%s' "$snapshot")" \
            "$state" "$(stat -c '%s' "$state")" >> "$MANIFEST"
    done
}

main() {
    load_checkpoints
    [[ -f "$DATASET" ]] || fail "Missing dataset: $DATASET"
    [[ -x "$TORCHRUN" ]] || fail "Missing executable torchrun: $TORCHRUN"

    local resume_state="" current_kimg=0 resume_args=()
    resume_state="$(latest_state || true)"
    if [[ -n "$resume_state" ]]; then
        current_kimg="$(state_kimg "$resume_state")"
        [[ -s "$(snapshot_path "$current_kimg")" ]] || fail "Latest state has no paired snapshot: $resume_state"
        resume_args+=(--resume="$resume_state")
    elif [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        fail "Run directory is non-empty but has no resumable state: $RUN_DIR"
    fi
    (( current_kimg <= TARGET_KIMG )) || fail "Latest state ${current_kimg} exceeds TARGET_KIMG=${TARGET_KIMG}"

    if [[ "$PLAN_ONLY" != "1" ]]; then
        source scripts/load_gpu_config.sh
        mkdir -p "$RUN_DIR" "$LOG_DIR" "$RESULTS_DIR"
    fi

    if (( current_kimg == TARGET_KIMG )); then
        log "Training already reaches ${TARGET_KIMG} kimg"
        if [[ "$PLAN_ONLY" != "1" ]]; then
            verify_outputs
            write_manifest
        fi
        return
    fi

    # A fixed seed would replay the stream from the start after a resume.
    local seed=$((SEED + current_kimg)) target checkpoint_csv=""
    for target in "${CHECKPOINT_KIMGS[@]}"; do
        (( target > current_kimg )) && checkpoint_csv+="${checkpoint_csv:+,}${target}"
    done

    log "Run directory: $RUN_DIR"
    log "Dataset subset: ${DATASET_MAX_SIZE} images selected with random seed ${DATASET_RANDOM_SEED}"
    log "Training range: ${current_kimg} -> ${TARGET_KIMG} kimg; seed: ${seed}"
    log "Checkpoints: $checkpoint_csv"
    if (( current_kimg > 0 )); then
        log "WARNING: resuming; this run is no longer a single continuous process"
    fi

    local duration command
    duration="$(awk -v kimg="$TARGET_KIMG" 'BEGIN { printf "%.6g", kimg / 1000 }')"
    command=(
        "$TORCHRUN" --standalone --nproc_per_node=1 train.py
        --outdir="$RUN_DIR" --nosubdir
        --data="$DATASET" --dataset-max-size="$DATASET_MAX_SIZE"
        --dataset-random-seed="$DATASET_RANDOM_SEED"
        --cond=1 --arch=ddpmpp --precond=edm
        --duration="$duration"
        --batch=512 --batch-gpu="$BATCH_GPU"
        --lr=0.001 --ema=0.5 --dropout=0.13 --augment=0.12
        --fp16=0 --seed="$seed"
        --tick="$TICK_KIMG" --snap=0 --dump=0
        --checkpoint-kimgs="$checkpoint_csv"
        "${resume_args[@]}"
    )

    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf '[PLAN] command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return
    fi

    "${command[@]}" 2>&1 | tee -a "$MASTER_LOG"
    verify_outputs
    write_manifest
    log "Training and checkpoint verification complete: $MANIFEST"
}

main "$@"
