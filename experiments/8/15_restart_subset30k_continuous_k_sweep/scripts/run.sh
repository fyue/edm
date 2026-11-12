#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
PYTHON="${PYTHON:-python}"
READ_ONLY=0
for arg in "$@"; do
    case "$arg" in --help|-h|--plan-only|--verify-only|--aggregate-only) READ_ONLY=1 ;; esac
done
if [[ "$READ_ONLY" != 1 && "${PLAN_ONLY:-0}" != 1 ]]; then
    source scripts/load_gpu_config.sh
fi
exec "$PYTHON" experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/watch_checkpoints.py "$@"
