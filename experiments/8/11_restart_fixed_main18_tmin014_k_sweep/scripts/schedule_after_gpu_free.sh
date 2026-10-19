#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
export EXP_ROOT="${EXP_ROOT:-experiments/8/11_restart_fixed_main18_tmin014_k_sweep}"
export RUN_SCRIPT=experiments/8/11_restart_fixed_main18_tmin014_k_sweep/scripts/run.sh
export SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 8-11 main18 tmin014 K sweep}"
# run.sh bypasses waiting for help, plan-only and CPU validation/aggregation.
READ_ONLY=0
EXPECT_EXP_ROOT=0
for arg in "$@"; do
    if [[ "$EXPECT_EXP_ROOT" == 1 ]]; then
        export EXP_ROOT="$arg"
        EXPECT_EXP_ROOT=0
    fi
    case "$arg" in
        --help|-h|--plan-only|--verify-only|--aggregate-only) READ_ONLY=1 ;;
        --exp-root) EXPECT_EXP_ROOT=1 ;;
        --exp-root=*) export EXP_ROOT="${arg#--exp-root=}" ;;
    esac
done
if [[ "$READ_ONLY" == 1 || "${PLAN_ONLY:-0}" == 1 || "${VERIFY_ONLY:-0}" == 1 || "${AGGREGATE_ONLY:-0}" == 1 ]]; then
    exec "$RUN_SCRIPT" "$@"
fi
export EDM_GPU_WAIT_DONE=1
exec scripts/wait_for_gpu_and_run.sh "$@"
