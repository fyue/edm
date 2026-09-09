#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../../../.."

PYTHON="${PYTHON:-python}"
PNDM_ROOT="experiments/4/pndm"
STAGE2="$PNDM_ROOT/stage2"
OUT="$STAGE2/2_fid50k_extra_seeds/results"

"$PYTHON" "$PNDM_ROOT/scripts/summarize_paired_seeds.py" \
    --summary "$STAGE2/1_fid50k_selected_churn/results/fid_summary.tsv" \
    --summary "$STAGE2/2_fid50k_extra_seeds/results/fid_summary.tsv" \
    --summary "$STAGE2/config/reused_experiment5_pretrained.tsv" \
    --pairs "$STAGE2/config/paired_conditions.tsv" \
    --condition-out "$OUT/fid50k_3seed_condition_stats.tsv" \
    --improvement-out "$OUT/fid50k_3seed_improvement_stats.tsv"
