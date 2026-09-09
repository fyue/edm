#!/usr/bin/env bash
set -Eeuo pipefail

PNDM_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

EXP_ROOT="experiments/4/pndm/stage1/1_fid5k_churn_sweep" \
CONFIG="experiments/4/pndm/stage1/1_fid5k_churn_sweep/config/conditions.tsv" \
NUM_IMAGES=5000 \
DEFAULT_SEEDS=0-4999 \
    "$PNDM_ROOT/scripts/run_fid_churn_sweep.sh"
