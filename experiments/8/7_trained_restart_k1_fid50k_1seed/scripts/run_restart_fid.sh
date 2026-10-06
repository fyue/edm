#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
exec "$PYTHON" experiments/8/7_trained_restart_k1_fid50k_1seed/scripts/run_experiment.py "$@"
