#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-4 FID-25k extension: reuse FID-5k images and generate 20k more.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/8/4_restart_k_sweep_10k_interval}"
export IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
export RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results/fid25k}"
export LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs/fid25k}"
export NUM_IMAGES="${NUM_IMAGES:-25000}"
export FID_SEEDS="${FID_SEEDS:-0-24999}"
export FID_GENERATION_SEEDS="${FID_GENERATION_SEEDS:-5000-24999}"
export REQUIRE_EXACT_IMAGE_COUNT="${REQUIRE_EXACT_IMAGE_COUNT:-1}"
export REQUIRE_EXACT_SEED_COVERAGE="${REQUIRE_EXACT_SEED_COVERAGE:-1}"
export SAMPLE_SEEDS="${SAMPLE_SEEDS:--}"
export EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-Experiment 8-4: Restart K=0..8 FID-25k with FID-5k reuse}"
export MASTER_LOG="${MASTER_LOG:-$LOG_DIR/restart-k-sweep-fid25k.log}"
export CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"
export CONDITION_FILTER="${CONDITION_FILTER:-}"

PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
CHECKPOINT_PROTOCOL="fid25k"
source "$EXP_ROOT/scripts/configure_checkpoints.sh"
REUSE_MODE="${REUSE_MODE:-hardlink}"
SOURCE_SUMMARY="$EXP_ROOT/results/restart_fid_summary.tsv"
REUSE_MANIFEST="$RESULTS_DIR/reuse_manifest.tsv"
BASE_RUNNER="experiments/8/1_restart_sampling_checkpoint_sweep/scripts/run_restart_checkpoint_sweep.sh"
RAW_SUMMARY="$RESULTS_DIR/restart_fid_summary.tsv"
SWEEP_SUMMARY="$RESULTS_DIR/restart_k_sweep.tsv"
OPTIMAL_SUMMARY="$RESULTS_DIR/optimal_k.tsv"

prepare_mode="materialize"
if [[ "${PLAN_ONLY:-0}" == "1" || "${SAMPLES_ONLY:-0}" == "1" ]]; then
    prepare_mode="check"
fi

"$PYTHON" "$EXP_ROOT/scripts/prepare_fid25k_reuse.py" \
    --config "$CONFIG" \
    --source-summary "$SOURCE_SUMMARY" \
    --image-root "$IMAGE_ROOT" \
    --manifest "$REUSE_MANIFEST" \
    --checkpoint-filter "$CHECKPOINT_FILTER" \
    --condition-filter "$CONDITION_FILTER" \
    --mode "$prepare_mode" \
    --reuse-mode "$REUSE_MODE"

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
        --optimal "$OPTIMAL_SUMMARY" \
        --output "$RESULTS_DIR/restart_k_sweep_overview"
    "$PYTHON" "$EXP_ROOT/scripts/create_analysis_views.py" \
        --sweep "$SWEEP_SUMMARY" \
        --table-output "$RESULTS_DIR/fid_by_checkpoint_and_k.tsv" \
        --markdown-output "$RESULTS_DIR/fid_by_checkpoint_and_k.md" \
        --panel-output "$RESULTS_DIR/fid_vs_k_by_checkpoint" \
        --heatmap-output "$RESULTS_DIR/fid_delta_vs_k0_heatmap"
else
    printf 'No complete K=0..8 FID-25k checkpoint sweep yet; skipping plots.\n'
fi
