#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 7-2のpilot runnerを起動する薄いwrapper。
# GPUの検出・待機・CUDA_VISIBLE_DEVICESの設定は共通scriptへ任せる。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/7/2_cifar10_cond_linear_fm_pilot}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_fm_linear_pilot.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 7-2 linear FM pilot}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
