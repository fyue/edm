#!/usr/bin/env bash
# CPU-only regression checks; never invoke training or generation.
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/run_train_and_fid.sh"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
PLAN_ONLY=1

train_command 10000
[[ " ${TRAIN_COMMAND[*]} " == *' --checkpoint-kimgs=2500,5000,7500,10000 '* ]]
train_command 30000 /tmp/training-state-020000.pt
[[ " ${TRAIN_COMMAND[*]} " == *' --checkpoint-kimgs=22500,25000,27500,30000 '* ]]
train_command 30000 /tmp/training-state-022500.pt
[[ " ${TRAIN_COMMAND[*]} " == *' --checkpoint-kimgs=25000,27500,30000 '* ]]

test_root="$(mktemp -d /tmp/edm-exp02-checkpoints.XXXXXX)"
RUN_DIR="$test_root/run"
RESULTS_DIR="$test_root/results"
mkdir -p "$RUN_DIR" "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/fid_summary.tsv"
printf '%s\n' '{"dataset_kwargs":{"max_size":30000,"random_seed":0,"use_labels":true},"seed":0,"batch_size":512}' \
    > "$RUN_DIR/training_options.json"

make_pair() {
    printf 'fixture\n' > "$(snapshot_path "$1")"
    printf 'fixture\n' > "$(state_path "$1")"
}
make_pair 10000
make_pair 20000
[[ "$(validate_existing_run)" == 20000 ]]
make_pair 22500
[[ "$(validate_existing_run)" == 22500 ]]

printf 'fixture\n' > "$(snapshot_path 25000)"
if (validate_existing_run) >/dev/null 2>&1; then
    printf 'Unpaired checkpoint was accepted\n' >&2
    exit 1
fi
printf 'fixture\n' > "$(state_path 25000)"
[[ "$(validate_existing_run)" == 25000 ]]

ensure_summary
printf '22500\t0-49999\t50000\t2.8\tsnapshot\timages\n' >> "$SUMMARY"
if (validate_summary_config) >/dev/null 2>&1; then
    printf 'FID at a non-10k checkpoint was accepted\n' >&2
    exit 1
fi

plan_output="$(show_plan)"
[[ "$(printf '%s\n' "$plan_output" | rg -c 'fid.py calc')" == 6 ]]
[[ "$(printf '%s\n' "$plan_output" | rg -c -- '--checkpoint-kimgs=')" == 6 ]]
printf 'Checkpoint scheduling, legacy run, intermediate resume, and FID cadence checks passed. Fixtures: %s\n' "$test_root"
