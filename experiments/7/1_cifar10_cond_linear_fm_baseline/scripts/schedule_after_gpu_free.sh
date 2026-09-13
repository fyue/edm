#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってから Experiment 7-1 の段階runnerを起動する薄いラッパー。
# 待機処理は共通の scripts/wait_for_gpu_and_run.sh に任せます。
#
#   PROFILE=smoke experiments/7/1_cifar10_cond_linear_fm_baseline/scripts/schedule_after_gpu_free.sh

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/7/1_cifar10_cond_linear_fm_baseline}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_fm_linear_baseline.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 7-1 (PROFILE=${PROFILE:-smoke})}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
