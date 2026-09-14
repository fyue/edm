#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-2を起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/2_restart_k5_fid5k_confirmation}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_restart_k5_fid5k.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-2 Selected Restart K FID-5k confirmation}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
