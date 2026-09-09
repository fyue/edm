#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the active 180k stage to finish and stop, then launch the separate
# three-seed evaluation of the saved ~175k snapshot on the same GPU.

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/3/5_170k-200k-lr2e-4-ema0.5}"
MASTER_LOG="$EXP_ROOT/logs/train-eval.log"
HANDOFF_LOG="$EXP_ROOT/logs/handoff-180k-to-175k.log"
SUMMARY="$EXP_ROOT/results/summary.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_STAGE"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$EXP_ROOT/logs"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$HANDOFF_LOG"
}

touch "$STOP_FILE"
log "Stop flag confirmed; waiting for the 180k third seed and stage shutdown"

while ! awk -F '\t' \
    'NR > 1 && $1 == 180000 && $2 == "100000-149999" { found = 1 } END { exit(found ? 0 : 1) }' \
    "$SUMMARY" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

while ! grep -Fq 'Found stop file; exiting after completed stage 180000 kimg' "$MASTER_LOG" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

log "180k stage stopped cleanly; starting the ~175k three-seed FID evaluation"
exec "$EXP_ROOT/scripts/run_fid50k_175k.sh"
