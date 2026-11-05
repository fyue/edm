#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/0/6_cifar10_cond_edm_30k_subset_continuous}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_training.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 0-6 EDM 30k subset continuous training}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
