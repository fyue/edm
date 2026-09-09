#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

# Compatibility wrapper for the documented Experiment 4 entry point.
PNDM_ROOT="experiments/4/pndm"
RUN_SCRIPT="${RUN_SCRIPT:-$PNDM_ROOT/scripts/run_all_stages.sh}"
EXP_ROOT="${EXP_ROOT:-$PNDM_ROOT}"
LOG_DIR="${LOG_DIR:-${SCHEDULE_LOG_DIR:-$PNDM_ROOT/logs}}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-Experiment 4 PNDM all stages}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME-Quadro RTX 8000}"

export RUN_SCRIPT EXP_ROOT LOG_DIR SCHEDULE_LABEL EXPECTED_GPU_NAME
exec scripts/wait_for_gpu_and_run.sh "$@"
