#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
RUN=experiments/0/4_post200k_restart_k1_fid50k_3seeds/scripts/run.sh
# Each phase uses the common GPU waiter through the CPU coordinator.
bash "$RUN" --checkpoint-filter 200k --samples-only
bash "$RUN"
