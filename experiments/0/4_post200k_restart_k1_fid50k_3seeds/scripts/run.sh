#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
exec "${PYTHON:-python}" experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run_experiment.py "$@"
