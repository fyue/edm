#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
source scripts/load_gpu_config.sh
exec "${PYTHON:-python}" experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run_experiment.py "$@"
