#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/load_gpu_config.sh"
EXP_ROOT="$REPO_ROOT/experiments/0/7_cifar10_cond_edm_post200k_no_replay"
exec "${PYTHON:-python}" -u "$EXP_ROOT/scripts/run_experiment.py" "$@"
