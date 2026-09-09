#!/usr/bin/env bash
set -Eeuo pipefail

PNDM_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

EXP_ROOT="experiments/4/pndm/stage2/1_fid50k_selected_churn" \
CONFIG="experiments/4/pndm/stage2/1_fid50k_selected_churn/config/selected_conditions.tsv" \
NUM_IMAGES=50000 \
DEFAULT_SEEDS=0-49999 \
    "$PNDM_ROOT/scripts/run_fid_churn_sweep.sh"
