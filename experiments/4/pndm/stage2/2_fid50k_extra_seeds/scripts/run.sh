#!/usr/bin/env bash
set -Eeuo pipefail

PNDM_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

EXP_ROOT="experiments/4/pndm/stage2/2_fid50k_extra_seeds" \
CONFIG="experiments/4/pndm/stage2/2_fid50k_extra_seeds/config/extra_seed_conditions.tsv" \
NUM_IMAGES=50000 \
DEFAULT_SEEDS=50000-99999 \
    "$PNDM_ROOT/scripts/run_fid_churn_sweep.sh"
