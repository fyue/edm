#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse the 0-2 runner with a separate 10k subset experiment and a 30k endpoint.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/5_cifar10_cond_edm_10k_subset}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-edm-seed0-subset10k}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/cifar10-cond-ddpmpp-edm-seed0-subset10k}"
DATASET_MAX_SIZE="${DATASET_MAX_SIZE:-10000}"
MAX_KIMG="${MAX_KIMG:-30000}"
EARLY_STOP="${EARLY_STOP:-0}"

SOURCE_SCRIPTS=experiments/0/2_cifar10_cond_edm_30k_subset/scripts
source "$SOURCE_SCRIPTS/run_train_and_fid.sh"
PLATEAU_SCRIPT="$SOURCE_SCRIPTS/evaluate_plateau.py"
IMAGE_POOL_SCRIPT="$SOURCE_SCRIPTS/inspect_image_pool.py"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
