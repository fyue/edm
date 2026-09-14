#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-1を起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/1_restart_sampling_checkpoint_sweep}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_restart_checkpoint_sweep.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-1 Restart checkpoint sweep}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
