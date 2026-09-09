#!/usr/bin/env bash
set -Eeuo pipefail

# Run only S_churn=6,7 in this experiment directory. The Stage 2 experiment 1
# runner supplies the shared FID-50k implementation through path overrides.

cd "$(dirname "$0")/../../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/4/stage2/3_fid50k_churn_6_7}"
CONFIG="${CONFIG:-$EXP_ROOT/config/churn_6_7.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
BASE_RUNNER="experiments/4/stage2/1_fid50k_selected_churn/scripts/run_fid50k_selected_churn.sh"
PLOT_SCRIPT="$EXP_ROOT/scripts/merge_and_plot_fid50k_churn.py"
PYTHON="${PYTHON:-python}"
PLOT_AFTER_RUN="${PLOT_AFTER_RUN:-1}"

EXP_ROOT="$EXP_ROOT" \
CONFIG="$CONFIG" \
IMAGE_ROOT="$IMAGE_ROOT" \
RESULTS_DIR="$RESULTS_DIR" \
LOG_DIR="$LOG_DIR" \
    "$BASE_RUNNER"

if [[ "${PLAN_ONLY:-0}" != "1" && "$PLOT_AFTER_RUN" == "1" ]]; then
    "$PYTHON" "$PLOT_SCRIPT"
fi
