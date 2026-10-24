#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
exec "${PYTHON:-python}" experiments/0/3_cifar10_cond_edm_post200k_extension/scripts/run_experiment.py "$@"
