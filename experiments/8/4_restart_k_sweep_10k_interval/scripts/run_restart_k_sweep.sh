#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-4 specialization of the Experiment 8-1 Restart runner.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/8/4_restart_k_sweep_10k_interval}"
export PAUSE_ENABLED="${PAUSE_ENABLED:-1}"
export NUM_IMAGES="${NUM_IMAGES:-5000}"
export FID_SEEDS="${FID_SEEDS:-0-4999}"
export SAMPLE_SEEDS="${SAMPLE_SEEDS:--}"
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-Experiment 8-4: Restart K=0..8 FID-5k at 10k intervals}"
export MASTER_LOG="${MASTER_LOG:-$EXP_ROOT/logs/restart-k-sweep-fid5k.log}"

PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
CHECKPOINT_PROTOCOL="fid5k"
source "$EXP_ROOT/scripts/configure_checkpoints.sh"
BASE_RUNNER="experiments/8/1_restart_sampling_checkpoint_sweep/scripts/run_restart_checkpoint_sweep.sh"
RAW_SUMMARY="$EXP_ROOT/results/restart_fid_summary.tsv"
SWEEP_SUMMARY="$EXP_ROOT/results/restart_k_sweep.tsv"
OPTIMAL_SUMMARY="$EXP_ROOT/results/optimal_k.tsv"

"$BASE_RUNNER"

if [[ "${PLAN_ONLY:-0}" == "1" || "${SAMPLES_ONLY:-0}" == "1" ]]; then
    exit 0
fi

"$PYTHON" "$EXP_ROOT/scripts/summarize_results.py" \
    --input "$RAW_SUMMARY" \
    --sweep-output "$SWEEP_SUMMARY" \
    --optimal-output "$OPTIMAL_SUMMARY" \
    --num-images "$NUM_IMAGES" \
    --seeds "$FID_SEEDS"

if awk -F '\t' 'NR > 1 { found = 1 } END { exit(found ? 0 : 1) }' "$OPTIMAL_SUMMARY"; then
    "$PYTHON" "$EXP_ROOT/scripts/plot_results.py" \
        --sweep "$SWEEP_SUMMARY" \
        --optimal "$OPTIMAL_SUMMARY"
    "$PYTHON" "$EXP_ROOT/scripts/create_analysis_views.py" \
        --sweep "$SWEEP_SUMMARY"
else
    printf 'No complete K=0..8 checkpoint sweep yet; skipping plots.\n'
fi
