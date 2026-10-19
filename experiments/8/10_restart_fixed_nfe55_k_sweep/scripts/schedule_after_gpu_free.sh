#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
export EXP_ROOT="${EXP_ROOT:-experiments/8/10_restart_fixed_nfe55_k_sweep}"
export RUN_SCRIPT=experiments/8/10_restart_fixed_nfe55_k_sweep/scripts/run.sh
export SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-10 fixed NFE55 K sweep excluding K5}"
# run.sh bypasses waiting for help, plan-only and CPU validation/aggregation.
READ_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --help|-h|--plan-only|--verify-only|--aggregate-only) READ_ONLY=1 ;;
    esac
done
if [[ "$READ_ONLY" == 1 || "${PLAN_ONLY:-0}" == 1 || "${VERIFY_ONLY:-0}" == 1 || "${AGGREGATE_ONLY:-0}" == 1 ]]; then
    exec "$RUN_SCRIPT" "$@"
fi
export EDM_GPU_WAIT_DONE=1
exec scripts/wait_for_gpu_and_run.sh "$@"
