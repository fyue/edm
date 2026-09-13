#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 7-4の評価runnerを起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/7/4_fm_edm_fid50k_confirmation}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_fid50k_confirmation.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 7-4 FM/EDM FID-50k confirmation}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
