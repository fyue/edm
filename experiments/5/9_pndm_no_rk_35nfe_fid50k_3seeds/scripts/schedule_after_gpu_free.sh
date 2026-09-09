#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/5/9_pndm_no_rk_35nfe_fid50k_3seeds}"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_pndm_no_rk_35nfe_fid50k_3seeds.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 5-9}"

export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL
exec scripts/wait_for_gpu_and_run.sh "$@"
