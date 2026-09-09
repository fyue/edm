#!/usr/bin/env bash
set -Eeuo pipefail

PNDM_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

EXP_ROOT="experiments/4/pndm/stage0/1_fid1k_churn_sweep" \
CONFIG="experiments/4/pndm/stage0/1_fid1k_churn_sweep/config/conditions.tsv" \
NUM_IMAGES=1024 \
DEFAULT_SEEDS=0-1023 \
    "$PNDM_ROOT/scripts/run_fid_churn_sweep.sh"
