#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../.."
EXP=experiments/8/14_restart_subset30k_k_sweep
PYTHON="${PYTHON:-python}"
unset EXP_ROOT
if [[ "${PLAN_ONLY:-0}" == 1 || "${1:-}" == --plan-only || "${1:-}" == --help ]]; then
    for mode in main18 nfe55; do
        "$EXP/scripts/run.sh" --mode "$mode" "${1:---plan-only}" --fid-stage all
    done
    exit 0
fi
if (( $# )); then
    echo 'run_all.sh accepts only --plan-only or --help; use run.sh for individual options' >&2
    exit 2
fi
# Each mode owns a separate output lock; this lock protects the full sequence.
mkdir -p "$EXP/logs"
exec 9>"$EXP/logs/.sequence.lock"
flock -n 9 || { echo 'Experiment sequence already running' >&2; exit 1; }
for mode in main18 nfe55; do
    "$EXP/scripts/run.sh" --mode "$mode" --samples-only --sample-seeds 0-1
done
for stage in 5000 25000 50000; do
    for mode in main18 nfe55; do
        "$EXP/scripts/run.sh" --mode "$mode" --fid-stage "$stage"
    done
    "$PYTHON" "$EXP/scripts/compare_results.py"
done
"$PYTHON" "$EXP/scripts/compare_results.py" --require-complete
