#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the saved 0-6 checkpoints after training, with the 0-2 FID protocol.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/6_cifar10_cond_edm_30k_subset_continuous}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/cifar10-cond-ddpmpp-edm-seed0-subset30k-continuous}"
MAX_KIMG=100000

SOURCE_SCRIPTS=experiments/0/2_cifar10_cond_edm_30k_subset/scripts
source "$SOURCE_SCRIPTS/run_train_and_fid.sh"
PLATEAU_SCRIPT="$SOURCE_SCRIPTS/evaluate_plateau.py"
IMAGE_POOL_SCRIPT="$SOURCE_SCRIPTS/inspect_image_pool.py"
MASTER_LOG="$LOG_DIR/fid.log"

main_fid() {
    validate_config
    local target
    if [[ "$PLAN_ONLY" == "1" ]]; then
        for ((target = FID_INTERVAL_KIMG; target <= MAX_KIMG; target += FID_INTERVAL_KIMG)); do
            print_command "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
                --outdir="$FID_ROOT/images-${target}-${FID_SEEDS}" --seeds="$FID_SEEDS" --subdirs \
                --sampler=edm --steps="$FID_STEPS" --S_churn=0 --network="$(snapshot_path "$target")"
            print_command "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
                --images="$FID_ROOT/images-${target}-${FID_SEEDS}" --ref="$FID_REF" --num="$FID_NUM_IMAGES"
        done
        return
    fi

    source scripts/load_gpu_config.sh
    mkdir -p "$FID_ROOT" "$LOG_DIR" "$RESULTS_DIR"
    ensure_summary
    validate_summary_config
    for ((target = FID_INTERVAL_KIMG; target <= MAX_KIMG; target += FID_INTERVAL_KIMG)); do
        [[ -s "$(snapshot_path "$target")" ]] || fail "Missing snapshot: $(snapshot_path "$target")"
        run_fid_for "$target"
    done
    log "FID evaluation complete: $SUMMARY"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main_fid "$@"
fi
