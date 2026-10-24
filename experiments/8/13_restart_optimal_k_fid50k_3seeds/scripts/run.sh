#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
# The CPU parent checks STOP before calling the common GPU waiter at each boundary.
exec "${PYTHON:-python}" experiments/8/13_restart_optimal_k_fid50k_3seeds/scripts/run_experiment.py "$@"
