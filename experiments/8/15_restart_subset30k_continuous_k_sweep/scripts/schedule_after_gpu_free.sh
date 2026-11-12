#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
export RUN_SCRIPT=experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/run_all.sh
export EXP_ROOT=experiments/8/15_restart_subset30k_continuous_k_sweep
export SCHEDULE_LABEL='Experiment 8-15 continuous subset30k'
exec scripts/wait_for_gpu_and_run.sh "$@"
