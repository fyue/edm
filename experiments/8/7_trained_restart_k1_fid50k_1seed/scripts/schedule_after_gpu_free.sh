#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/8/7_trained_restart_k1_fid50k_1seed}"
RUN_SCRIPT="${RUN_SCRIPT:-experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/run_restart_fid.sh}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-7 trained Restart K=1 vs ODE, 39 NFE, FID-50k}"
export EXP_ROOT RUN_SCRIPT SCHEDULE_LABEL

# Help and plan inspection do not need a GPU allocation.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "$RUN_SCRIPT" "$@"
fi
exec scripts/wait_for_gpu_and_run.sh "$@"
