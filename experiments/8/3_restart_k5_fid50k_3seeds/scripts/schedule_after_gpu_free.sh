#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-3を起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/3_restart_k5_fid50k_3seeds}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_restart_k5_fid50k_3seeds.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-3 Restart K=5 FID-50k x 3 seed ranges}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
