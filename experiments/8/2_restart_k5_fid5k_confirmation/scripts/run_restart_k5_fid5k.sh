#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-2 specialization of the Experiment 8-1 Restart runner.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/8/2_restart_k5_fid5k_confirmation}"
export CONFIG="${CONFIG:-$EXP_ROOT/config/restart_k5_fid5k.tsv}"
export NUM_IMAGES="${NUM_IMAGES:-5000}"
export FID_SEEDS="${FID_SEEDS:-0-4999}"
export SAMPLE_SEEDS="${SAMPLE_SEEDS:--}"
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-Experiment 8-2: Selected Restart K FID-5k confirmation}"
export MASTER_LOG="${MASTER_LOG:-$EXP_ROOT/logs/restart-k5-fid5k.log}"

PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
BASE_RUNNER="experiments/8/1_restart_sampling_checkpoint_sweep/scripts/run_restart_checkpoint_sweep.sh"
SUMMARY="$EXP_ROOT/results/restart_fid_summary.tsv"
IMPROVEMENTS="$EXP_ROOT/results/restart_improvements.tsv"

"$BASE_RUNNER"

if [[ "${PLAN_ONLY:-0}" == "1" || "${SAMPLES_ONLY:-0}" == "1" ]]; then
    exit 0
fi

"$PYTHON" experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py \
    --input "$SUMMARY" \
    --output "$IMPROVEMENTS"
if awk -F '\t' 'NR > 1 && $8 != "NA" && $10 != "NA" { found = 1 } END { exit(found ? 0 : 1) }' \
    "$IMPROVEMENTS"; then
    "$PYTHON" "$EXP_ROOT/scripts/plot_results.py" \
        --input "$IMPROVEMENTS" \
        --num-images "$NUM_IMAGES" \
        --seeds "$FID_SEEDS"
else
    printf 'No complete Restart/baseline triplet yet; skipping plot.\n'
fi
