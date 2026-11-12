#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
EXP_ROOT="$REPO_ROOT/experiments/0/7_cifar10_cond_edm_post200k_no_replay"
for arg in "$@"; do
    case "$arg" in
        --help|-h|--plan-only|--dry-run|--verify-only)
            exec "${PYTHON:-python}" "$EXP_ROOT/scripts/run_experiment.py" "$@" ;;
    esac
done
if [[ "${PLAN_ONLY:-0}" == 1 ]]; then
    exec "${PYTHON:-python}" "$EXP_ROOT/scripts/run_experiment.py" --plan-only "$@"
fi
export EXP_ROOT
export RUN_SCRIPT="$EXP_ROOT/scripts/gpu_worker.sh"
export SCHEDULE_LABEL="0-7 200k to 300k training"
exec "$REPO_ROOT/scripts/wait_for_gpu_and_run.sh" "$@"
