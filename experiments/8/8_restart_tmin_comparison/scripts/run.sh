#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
SCRIPT_DIR=experiments/8/8_restart_tmin_comparison/scripts
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
READ_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --help|-h|--plan-only|--verify-only|--aggregate-only) READ_ONLY=1 ;;
    esac
done
if [[ "$READ_ONLY" == 1 || "${PLAN_ONLY:-0}" == 1 || "${VERIFY_ONLY:-0}" == 1 || "${AGGREGATE_ONLY:-0}" == 1 ]]; then
    exec "$PYTHON" "$SCRIPT_DIR/run_experiment.py" "$@"
fi
source scripts/load_gpu_config.sh
if [[ "${EDM_GPU_WAIT_DONE:-0}" == 1 ]]; then
    if [[ "${CUDA_VISIBLE_DEVICES:-}" != "$GPU_INDEX" || "${CUDA_DEVICE_ORDER:-}" != PCI_BUS_ID ]]; then
        printf 'GPU policy: CUDA settings must match the local GPU configuration.\n' >&2
        exit 1
    fi
    exec "$PYTHON" "$SCRIPT_DIR/run_experiment.py" "$@"
fi
exec "$SCRIPT_DIR/schedule_after_gpu_free.sh" "$@"
