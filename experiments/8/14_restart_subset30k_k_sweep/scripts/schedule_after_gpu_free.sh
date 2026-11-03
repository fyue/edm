#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
export RUN_SCRIPT=experiments/8/14_restart_subset30k_k_sweep/scripts/run_all.sh
export EXP_ROOT=experiments/8/14_restart_subset30k_k_sweep
export SCHEDULE_LABEL='Experiment 8-14 subset30k 20k/40k/60k'
# run_all.sh clears EXP_ROOT before choosing mode-specific output roots.
exec scripts/wait_for_gpu_and_run.sh "$@"
