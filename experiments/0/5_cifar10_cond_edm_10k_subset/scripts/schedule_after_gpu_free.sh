#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/5_cifar10_cond_edm_10k_subset}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_train_and_fid.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 0-5 EDM 10k subset train to 30k and FID}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
