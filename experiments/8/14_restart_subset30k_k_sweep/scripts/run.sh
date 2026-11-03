#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
SCRIPT_DIR=experiments/8/14_restart_subset30k_k_sweep/scripts
PYTHON="${PYTHON:-python}"
# --mode is handled here; the Python runner uses FIXED_MODE.
if [[ "${1:-}" == "--mode" ]]; then
    export FIXED_MODE="${2:?--mode requires main18 or nfe55}"
    shift 2
fi
export FIXED_MODE="${FIXED_MODE:-main18}"
if [[ "$FIXED_MODE" != main18 && "$FIXED_MODE" != nfe55 ]]; then
    echo "Invalid --mode: $FIXED_MODE" >&2
    exit 2
fi
READ_ONLY=0
EXPECT_EXP_ROOT=0
for arg in "$@"; do
    if [[ "$EXPECT_EXP_ROOT" == 1 ]]; then
        export EXP_ROOT="$arg"
        EXPECT_EXP_ROOT=0
    fi
    case "$arg" in
        --help|-h|--plan-only|--verify-only|--aggregate-only|--register-runner-compatibility) READ_ONLY=1 ;;
        --exp-root) EXPECT_EXP_ROOT=1 ;;
        --exp-root=*) export EXP_ROOT="${arg#--exp-root=}" ;;
    esac
done
if [[ "$READ_ONLY" == 1 || "${PLAN_ONLY:-0}" == 1 || "${VERIFY_ONLY:-0}" == 1 || "${AGGREGATE_ONLY:-0}" == 1 ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/run_experiment.py" "$@"
fi
source scripts/load_gpu_config.sh
export EDM_GPU_WAIT_DONE=1
exec "$PYTHON" "$SCRIPT_DIR/run_experiment.py" "$@"
