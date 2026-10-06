#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 0-2の学習とFID評価を起動するwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/2_cifar10_cond_edm_30k_subset}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_train_and_fid.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 0-2 EDM 30k subset train and FID early stop}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
