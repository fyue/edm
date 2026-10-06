#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-6の指定FID stageを起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/6_exact_10k_restart_k5_fid50k_1seed}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_progressive_fid.sh}"
FID_STAGE="${FID_STAGE:-5000}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-6 progressive FID stage $FID_STAGE}"

export EXP_ROOT RUN_SCRIPT FID_STAGE SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
