#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 0-1の再学習を起動する薄いwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/1_cifar10_cond_edm_10k_retraining}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_training.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 0-1 EDM exact 10k retraining}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
