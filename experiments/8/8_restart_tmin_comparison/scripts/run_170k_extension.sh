#!/usr/bin/env bash
# Run inside the existing edm-exp8-8 tmux session. GPU work always uses run.sh.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
SCRIPT_DIR=experiments/8/8_restart_tmin_comparison/scripts
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
source scripts/load_gpu_config.sh
if [[ -z "${TMUX:-}" || -z "${TMUX_PANE:-}" ]]; then
    printf 'Run this workflow inside tmux session edm-exp8-8.\n' >&2
    exit 1
fi
if [[ "$(tmux display-message -p -t "$TMUX_PANE" '#S')" != edm-exp8-8 ]]; then
    printf 'Run this workflow inside tmux session edm-exp8-8.\n' >&2
    exit 1
fi
export CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
"$SCRIPT_DIR/run.sh" --checkpoint-filter 170k --samples-only --sample-seeds 0-1 --resume
# A second pass verifies smoke resume without regenerating either seed.
"$SCRIPT_DIR/run.sh" --checkpoint-filter 170k --samples-only --sample-seeds 0-1 --resume
tmux new-window -d -t edm-exp8-8 -n completion -c "$PWD" \
    "$PYTHON $SCRIPT_DIR/watch_completion.py --interval 30 --wait-for-run --runner-pid $$"
exec "$SCRIPT_DIR/run.sh" --checkpoint-filter 170k --fid-stage 50000 --resume
