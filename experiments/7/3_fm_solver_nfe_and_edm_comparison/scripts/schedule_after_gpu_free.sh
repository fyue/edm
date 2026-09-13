#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 7-3の評価runnerを起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/7/3_fm_solver_nfe_and_edm_comparison}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_fm_solver_nfe_comparison.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 7-3 FM solver/NFE comparison}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
