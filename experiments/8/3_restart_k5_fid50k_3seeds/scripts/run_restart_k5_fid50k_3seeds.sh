#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-3: run the fixed K=5 comparison over three FID-50k seed ranges.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/8/3_restart_k5_fid50k_3seeds}"
export CONFIG="${CONFIG:-$EXP_ROOT/config/restart_k5_fid50k.tsv}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export SAMPLE_SEEDS="${SAMPLE_SEEDS:--}"
export MASTER_LOG="${MASTER_LOG:-$EXP_ROOT/logs/restart-k5-fid50k-3seeds.log}"

FID_SEEDS_LIST="${FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
SEED_FILTER="${SEED_FILTER:-}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
BASE_RUNNER="experiments/8/1_restart_sampling_checkpoint_sweep/scripts/run_restart_checkpoint_sweep.sh"
SUMMARY="$EXP_ROOT/results/restart_fid_summary.tsv"
IMPROVEMENTS="$EXP_ROOT/results/restart_improvements.tsv"
STATS="$EXP_ROOT/results/restart_k5_fid50k_3seed_stats.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/results"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

matches_filter() {
    local value="$1"
    local filter="$2"
    local wanted
    if [[ -z "$filter" ]]; then
        return 0
    fi
    for wanted in $filter; do
        if [[ "$wanted" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

validate_seed_range() {
    local seeds="$1"
    local start
    local end
    if [[ ! "$seeds" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        log "Invalid seed range: $seeds"
        return 1
    fi
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    if (( end < start || end - start + 1 != NUM_IMAGES )); then
        log "Seed range $seeds does not contain NUM_IMAGES=$NUM_IMAGES seeds"
        return 1
    fi
}

aggregate_results() {
    if [[ ! -f "$SUMMARY" ]]; then
        log "No result summary yet; skipping aggregation"
        return 0
    fi
    "$PYTHON" experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py \
        --input "$SUMMARY" \
        --output "$IMPROVEMENTS"
    "$PYTHON" "$EXP_ROOT/scripts/summarize_3seeds.py" \
        --input "$IMPROVEMENTS" \
        --output "$STATS" \
        --num-images "$NUM_IMAGES" \
        --expected-seeds "$FID_SEEDS_LIST"
    if awk -F '\t' 'NR > 1 && $NF == "1" { found = 1 } END { exit(found ? 0 : 1) }' \
        "$STATS"; then
        "$PYTHON" "$EXP_ROOT/scripts/plot_results.py" \
            --input "$STATS" \
            --num-images "$NUM_IMAGES"
    else
        log "No checkpoint has all expected seed ranges yet; skipping plot"
    fi
}

main() {
    local seeds
    local selected=0
    local -A seen=()

    if [[ ! "$NUM_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
        log "NUM_IMAGES must be a positive integer: $NUM_IMAGES"
        exit 1
    fi
    if [[ ! -x "$PYTHON" ]]; then
        log "Missing executable Python: $PYTHON"
        exit 1
    fi
    for seeds in $FID_SEEDS_LIST; do
        validate_seed_range "$seeds"
        if [[ -n "${seen[$seeds]:-}" ]]; then
            log "Duplicate seed range in FID_SEEDS_LIST: $seeds"
            exit 1
        fi
        seen[$seeds]=1
    done

    log "Experiment 8-3: Restart K=5 FID-$NUM_IMAGES over seed ranges"
    log "Seed ranges: $FID_SEEDS_LIST"
    log "Seed filter: ${SEED_FILTER:-(all)}"

    for seeds in $FID_SEEDS_LIST; do
        if ! matches_filter "$seeds" "$SEED_FILTER"; then
            log "[seeds=$seeds] Filtered out; skipping"
            continue
        fi
        selected=$((selected + 1))
        export FID_SEEDS="$seeds"
        export EXPERIMENT_LABEL="Experiment 8-3: Restart K=5 FID-$NUM_IMAGES seeds $seeds"
        "$BASE_RUNNER"
        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; not starting another seed range"
            break
        fi
    done

    if (( selected == 0 )); then
        log "No seed ranges matched SEED_FILTER"
        exit 1
    fi
    if [[ "${PLAN_ONLY:-0}" == "1" || "${SAMPLES_ONLY:-0}" == "1" ]]; then
        exit 0
    fi

    aggregate_results
    log "Experiment 8-3 runner finished"
}

main "$@"
