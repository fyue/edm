#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/5/13_trajectory_distance_5-4_vs_5-11}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_trajectory_distance.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 5-13 trajectory distance}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
