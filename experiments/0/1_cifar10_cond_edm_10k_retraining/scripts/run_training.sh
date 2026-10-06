#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 0-1: retrain the CIFAR-10 conditional EDM baseline in one process
# and save an exact, uniform 10k checkpoint series through 200k.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"

EXP_ROOT="${EXP_ROOT:-experiments/0/1_cifar10_cond_edm_10k_retraining}"
CHECKPOINT_CONFIG="${CHECKPOINT_CONFIG:-$EXP_ROOT/config/checkpoint_kimgs.txt}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-edm-seed0-10k}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
TARGET_KIMG="${TARGET_KIMG:-200000}"
TICK_KIMG="${TICK_KIMG:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
SEED="${SEED:-0}"
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

load_and_validate_checkpoints() {
    if [[ ! -f "$CHECKPOINT_CONFIG" ]]; then
        printf 'Missing checkpoint config: %s\n' "$CHECKPOINT_CONFIG" >&2
        return 1
    fi
    mapfile -t CHECKPOINT_KIMGS < <(awk '!/^#/ && NF { print $1 }' "$CHECKPOINT_CONFIG")
    if (( ${#CHECKPOINT_KIMGS[@]} == 0 )); then
        printf 'No checkpoints in %s\n' "$CHECKPOINT_CONFIG" >&2
        return 1
    fi

    local index expected target
    for index in "${!CHECKPOINT_KIMGS[@]}"; do
        target="${CHECKPOINT_KIMGS[$index]}"
        expected=$(( (index + 1) * 10000 ))
        if [[ ! "$target" =~ ^[0-9]+$ ]] || (( target != expected )); then
            printf 'Expected checkpoint %d at line %d, got %s\n' "$expected" "$((index + 1))" "$target" >&2
            return 1
        fi
    done
    if (( CHECKPOINT_KIMGS[${#CHECKPOINT_KIMGS[@]} - 1] != TARGET_KIMG )); then
        printf 'Final checkpoint must equal TARGET_KIMG=%d\n' "$TARGET_KIMG" >&2
        return 1
    fi
    CHECKPOINT_CSV="$(IFS=,; printf '%s' "${CHECKPOINT_KIMGS[*]}")"
}

verify_outputs() {
    local missing=0 target snapshot state
    for target in "${CHECKPOINT_KIMGS[@]}"; do
        snapshot="$(snapshot_path "$target")"
        state="$(state_path "$target")"
        if [[ ! -s "$snapshot" ]]; then
            log "Missing snapshot: $snapshot"
            missing=1
        fi
        if [[ ! -s "$state" ]]; then
            log "Missing state: $state"
            missing=1
        fi
    done
    (( missing == 0 ))
}

write_manifest() {
    local target snapshot state
    mkdir -p "$RESULTS_DIR"
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
    load_and_validate_checkpoints

    if [[ ! -f "$DATASET" ]]; then
        printf 'Missing dataset: %s\n' "$DATASET" >&2
        return 1
    fi
    if [[ ! -x "$TORCHRUN" ]]; then
        printf 'Missing executable torchrun: %s\n' "$TORCHRUN" >&2
        return 1
    fi
    if [[ "$PLAN_ONLY" != "1" ]]; then
        mkdir -p "$RUN_DIR" "$LOG_DIR" "$RESULTS_DIR"
    fi

    local resume_state=""
    local current_kimg=0
    local resume_args=()
    resume_state="$(latest_state || true)"
    if [[ -n "$resume_state" ]]; then
        current_kimg="$(state_kimg "$resume_state")"
        if [[ ! -s "$(snapshot_path "$current_kimg")" ]]; then
            printf 'Latest state has no paired snapshot: %s\n' "$resume_state" >&2
            return 1
        fi
        resume_args+=(--resume="$resume_state")
    elif [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        printf 'Run directory is non-empty but has no resumable state: %s\n' "$RUN_DIR" >&2
        return 1
    fi

    if (( current_kimg > TARGET_KIMG )); then
        printf 'Latest state %d exceeds TARGET_KIMG=%d\n' "$current_kimg" "$TARGET_KIMG" >&2
        return 1
    fi

    log "Experiment root: $EXP_ROOT"
    log "Run directory: $RUN_DIR"
    log "Training range: ${current_kimg} -> ${TARGET_KIMG} kimg"
    log "Exact checkpoint labels: $CHECKPOINT_CSV"
    log "Regular snapshot/state intervals: disabled"
    log "Seed: $SEED; batch: 512; batch-gpu: $BATCH_GPU; GPU: ${CUDA_VISIBLE_DEVICES}"

    if (( current_kimg == TARGET_KIMG )); then
        if [[ "$PLAN_ONLY" == "1" ]]; then
            log "Training already reaches ${TARGET_KIMG} kimg; verify 20 checkpoint pairs"
            return 0
        fi
        verify_outputs
        write_manifest
        log "All checkpoint pairs already exist; manifest refreshed: $MANIFEST"
        return 0
    fi

    local duration
    duration="$(awk -v kimg="$TARGET_KIMG" 'BEGIN { printf "%.6g", kimg / 1000 }')"
    local command=(
        "$TORCHRUN" --standalone --nproc_per_node=1 train.py
        --outdir="$RUN_DIR" --nosubdir
        --data="$DATASET"
        --cond=1 --arch=ddpmpp --precond=edm
        --duration="$duration"
        --batch=512 --batch-gpu="$BATCH_GPU"
        --lr=0.001 --ema=0.5 --dropout=0.13 --augment=0.12
        --fp16=0 --seed="$SEED"
        --tick="$TICK_KIMG" --snap=0 --dump=0
        --checkpoint-kimgs="$CHECKPOINT_CSV"
        "${resume_args[@]}"
    )

    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf 'PLAN command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    "${command[@]}" 2>&1 | tee -a "$MASTER_LOG"
    verify_outputs
    write_manifest
    log "Training and checkpoint verification complete: $MANIFEST"
}

main "$@"
