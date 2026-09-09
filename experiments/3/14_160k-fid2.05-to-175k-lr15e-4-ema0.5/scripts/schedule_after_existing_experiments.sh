#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for currently running EDM experiment commands to finish, then launch
# Experiment 3-14. This is intentionally process-based so it also works when
# the existing run was started outside tmux.

cd "$(dirname "$0")/../../../.."

EXP_ROOT="${EXP_ROOT:-experiments/3/14_160k-fid2.05-to-175k-lr15e-4-ema0.5}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
WAIT_LOG="$LOG_DIR/schedule-after-existing.log"
RUN_SCRIPT="${RUN_SCRIPT:-$EXP_ROOT/scripts/run_160k_fid2.05_to_175k_lr15e-4_then_eval_all.sh}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"

mkdir -p "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$WAIT_LOG"
}

active_existing_commands() {
    pgrep -af 'torchrun|train.py|generate.py|fid.py|experiments/3/.*/scripts/run_.*\.sh' \
        | grep -v "$$" \
        | grep -v "$RUN_SCRIPT" \
        | grep -v 'schedule_after_existing_experiments.sh' \
        | grep -v 'pgrep -af' || true
}

main() {
    local active

    log "Scheduling Experiment 3-14 after existing EDM experiment commands complete"
    log "Run script: $RUN_SCRIPT"
    log "Check interval: ${CHECK_INTERVAL_SEC}s"

    while true; do
        active="$(active_existing_commands)"
        if [[ -z "$active" ]]; then
            break
        fi
        log "Existing experiment commands are still active:"
        printf '%s\n' "$active" | tee -a "$WAIT_LOG"
        sleep "$CHECK_INTERVAL_SEC"
    done

    log "No active existing experiment commands found; launching Experiment 3-14"
    exec "$RUN_SCRIPT"
}

main "$@"
