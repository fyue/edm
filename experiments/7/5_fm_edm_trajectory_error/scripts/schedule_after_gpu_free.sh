#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/7/5_fm_edm_trajectory_error}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_trajectory_error.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 7-5 FM/EDM trajectory error}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
