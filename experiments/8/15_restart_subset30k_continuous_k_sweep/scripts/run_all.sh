#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
exec experiments/8/15_restart_subset30k_continuous_k_sweep/scripts/run.sh --mode all "$@"
