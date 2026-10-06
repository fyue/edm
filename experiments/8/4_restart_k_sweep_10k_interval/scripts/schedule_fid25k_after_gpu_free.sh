#!/usr/bin/env bash
set -Eeuo pipefail

# GPUが空くまで待ってからExperiment 8-4 FID-25k extensionを起動するwrapper。

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/4_restart_k_sweep_10k_interval}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_restart_k_sweep_fid25k.sh}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs/fid25k}"
WAIT_LOG="${WAIT_LOG:-$LOG_DIR/schedule-after-gpu-free.log}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-4 Restart K=0..8 FID-25k with FID-5k reuse}"

export EXP_ROOT RUN_SCRIPT LOG_DIR WAIT_LOG SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
