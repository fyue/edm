#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-4を起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/4_restart_k_sweep_10k_interval}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_restart_k_sweep.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-4 Restart K=0..8 FID-5k at 10k intervals}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
