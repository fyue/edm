#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
export EXP_ROOT="${EXP_ROOT:-experiments/8/13_restart_optimal_k_fid50k_3seeds}"
export RUN_SCRIPT=experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/run.sh
export SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-13 optimal K FID50k 3seeds}"
# run.sh checks STOP first, then uses scripts/wait_for_gpu_and_run.sh.
exec "$RUN_SCRIPT" "$@"
