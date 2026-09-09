#!/usr/bin/env bash
set -Eeuo pipefail

# Run only the new S_churn=6,7 conditions in an isolated experiment folder.
# The Stage 1 experiment 1 runner is reused through its path overrides; its
# existing config, images, logs, and results are never modified.

cd "$(dirname "$0")/../../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/4/stage1/2_fid5k_churn_6_7}"
CONFIG="${CONFIG:-$EXP_ROOT/config/churn_6_7.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
BASE_RUNNER="experiments/4/stage1/1_fid5k_selected_churn/scripts/run_fid5k_selected_churn.sh"
PLOT_SCRIPT="$EXP_ROOT/scripts/merge_and_plot_fid5k_churn.py"
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
