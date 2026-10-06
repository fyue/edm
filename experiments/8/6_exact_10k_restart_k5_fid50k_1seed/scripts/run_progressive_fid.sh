#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-6: progressively extend each condition's image pool to 5k,
# 25k, and optionally 50k, evaluating FID after each requested stage.

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"
EXP_ROOT="${EXP_ROOT:-experiments/8/6_exact_10k_restart_k5_fid50k_1seed}"
CONFIG="${CONFIG:-$EXP_ROOT/config/restart_k5_fid50k.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_STAGE="${FID_STAGE:-5000}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:--}"
SAMPLES_ONLY="${SAMPLES_ONLY:-0}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
RHO="${RHO:-7}"
RESTART_NOISE="${RESTART_NOISE:-1}"
CONDITION_FILTER="${CONDITION_FILTER:-}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="${MASTER_LOG:-$LOG_DIR/progressive-fid.log}"
SUMMARY="$RESULTS_DIR/restart_fid_summary.tsv"
IMPROVEMENTS="$RESULTS_DIR/restart_improvements.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"
POOL_VALIDATOR="$EXP_ROOT/scripts/validate_image_pool.py"

PREVIOUS_STAGE=""
DELTA_SEEDS=""
FID_SEEDS=""

mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

configure_stage() {
    case "$FID_STAGE" in
        5000)
            PREVIOUS_STAGE=0
            DELTA_SEEDS="0-4999"
            ;;
        25000)
            PREVIOUS_STAGE=5000
            DELTA_SEEDS="5000-24999"
            ;;
        50000)
            PREVIOUS_STAGE=25000
            DELTA_SEEDS="25000-49999"
            ;;
        *)
            printf 'FID_STAGE must be one of 5000, 25000, or 50000: %s\n' "$FID_STAGE" >&2
            return 1
            ;;
    esac
    FID_SEEDS="0-$((FID_STAGE - 1))"
}

tokenize() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/\./p/g'
}

count_pngs() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        printf '0'
        return
    fi
    find "$dir" -type f -name '*.png' | wc -l | tr -d ' '
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

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'condition\tcheckpoint\tmethod\tmain_steps\trestart_steps\trestart_k\tt_min\tt_max\tnfe\tcomparison_group\tnum_images\tseeds\tfid\tgenerate_sec\tfid_sec\tnetwork\timages\tlog\tfid_ref\n' > "$SUMMARY"
    fi
}

result_done() {
    local condition="$1"
    local num_images="$2"
    local seeds="$3"
    if [[ ! -f "$SUMMARY" ]]; then
        return 1
    fi
    awk -F '\t' -v condition="$condition" -v num="$num_images" \
        -v seeds="$seeds" -v ref="$FID_REF" \
        'NR > 1 && $1 == condition && $11 == num && $12 == seeds && $19 == ref { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

expected_nfe() {
    local main_steps="$1"
    local restart_steps="$2"
    local restart_k="$3"
    local extra=0
    if (( restart_k > 0 )); then
        extra="$((restart_k * 2 * (restart_steps - 1)))"
    fi
    printf '%s' "$((2 * main_steps - 1 + extra))"
}

validate_condition() {
    local condition="$1"
    local main_steps="$2"
    local restart_steps="$3"
    local restart_k="$4"
    local t_min="$5"
    local t_max="$6"
    local nfe="$7"
    local network="$8"
    local calculated

    if [[ ! "$main_steps" =~ ^[0-9]+$ || ! "$restart_steps" =~ ^[0-9]+$ || ! "$restart_k" =~ ^[0-9]+$ || ! "$nfe" =~ ^[1-9][0-9]*$ ]]; then
        log "[$condition] invalid integer field"
        return 1
    fi
    if (( main_steps < 2 )); then
        log "[$condition] main_steps must be >= 2"
        return 1
    fi
    if (( restart_k > 0 )); then
        if (( restart_steps < 2 )); then
            log "[$condition] restart_steps must be >= 2 when K > 0"
            return 1
        fi
        if ! awk -v lo="$t_min" -v hi="$t_max" 'BEGIN { exit(!(lo > 0 && hi > lo)) }'; then
            log "[$condition] require 0 < t_min < t_max"
            return 1
        fi
    elif (( restart_steps != 0 )); then
        log "[$condition] ordinary ODE rows must use restart_steps=0 and restart_k=0"
        return 1
    fi
    calculated="$(expected_nfe "$main_steps" "$restart_steps" "$restart_k")"
    if [[ "$calculated" != "$nfe" ]]; then
        log "[$condition] configured NFE=$nfe but expected $calculated"
        return 1
    fi
    if [[ ! -f "$network" ]]; then
        log "[$condition] missing checkpoint: $network"
        return 1
    fi
}

check_inputs() {
    local failed=0
    if [[ ! -f "$CONFIG" ]]; then
        log "Missing config: $CONFIG"
        failed=1
    fi
    if [[ ! -f "$FID_REF" ]]; then
        log "Missing FID reference: $FID_REF"
        failed=1
    fi
    if [[ ! -x "$PYTHON" ]]; then
        log "Missing executable Python: $PYTHON"
        failed=1
    fi
    if [[ ! -f "$POOL_VALIDATOR" ]]; then
        log "Missing pool validator: $POOL_VALIDATOR"
        failed=1
    fi
    if [[ "$PLAN_ONLY" != "1" ]] && ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing torchrun: $TORCHRUN"
        failed=1
    fi
    if (( failed != 0 )); then
        exit 1
    fi
}

restart_json() {
    local restart_steps="$1"
    local restart_k="$2"
    local t_min="$3"
    local t_max="$4"
    if (( restart_k == 0 )); then
        printf '{}'
    else
        printf '{"0":[%s,%s,%s,%s]}' "$restart_steps" "$restart_k" "$t_min" "$t_max"
    fi
}

generate_images() {
    local network="$1"
    local outdir="$2"
    local seeds="$3"
    local main_steps="$4"
    local info="$5"
    local run_log="$6"
    local subdirs="$7"
    local extra_args=()
    if [[ "$subdirs" == "1" ]]; then
        extra_args+=(--subdirs)
    fi

    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --network="$network" \
        --outdir="$outdir" \
        --seeds="$seeds" \
        --batch="$GEN_BATCH" \
        --sampler=restart \
        --steps="$main_steps" \
        --rho="$RHO" \
        --S_churn=0 \
        --restart_info="$info" \
        --restart_noise="$RESTART_NOISE" \
        "${extra_args[@]}" 2>&1 | tee -a "$run_log"
}

generate_samples() {
    local condition="$1"
    local network="$2"
    local main_steps="$3"
    local info="$4"
    local sample_dir="$IMAGE_ROOT/$condition/samples-seeds-$(tokenize "$SAMPLE_SEEDS")"
    local marker="$sample_dir/.complete"
    local run_log="$LOG_DIR/$condition-samples.log"

    if [[ "$SAMPLE_SEEDS" == "-" ]]; then
        return 0
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition] fixed-seed samples $SAMPLE_SEEDS -> $sample_dir"
        return 0
    fi
    if [[ -f "$marker" ]]; then
        log "[$condition] fixed-seed samples already complete; skipping"
        return 0
    fi
    mkdir -p "$sample_dir"
    generate_images "$network" "$sample_dir" "$SAMPLE_SEEDS" "$main_steps" "$info" "$run_log" 0
    touch "$marker"
}

require_previous_stage() {
    local condition="$1"
    local previous_seeds
    if (( PREVIOUS_STAGE == 0 )); then
        return 0
    fi
    previous_seeds="0-$((PREVIOUS_STAGE - 1))"
    if ! result_done "$condition" "$PREVIOUS_STAGE" "$previous_seeds"; then
        log "[$condition] missing prerequisite FID-$PREVIOUS_STAGE result for seeds $previous_seeds"
        log "[$condition] run FID_STAGE=$PREVIOUS_STAGE first with the same filters"
        return 1
    fi
}

run_condition() {
    local condition="$1"
    local checkpoint="$2"
    local main_steps="$3"
    local restart_steps="$4"
    local restart_k="$5"
    local t_min="$6"
    local t_max="$7"
    local nfe="$8"
    local comparison_group="$9"
    local network="${10}"
    local method="ode-heun"
    local info
    local pool
    local run_log
    local png_count
    local generate_start
    local generate_sec="NA"
    local fid_start
    local fid_sec
    local fid

    validate_condition "$condition" "$main_steps" "$restart_steps" "$restart_k" "$t_min" "$t_max" "$nfe" "$network"
    if (( restart_k > 0 )); then
        method="restart"
    fi
    info="$(restart_json "$restart_steps" "$restart_k" "$t_min" "$t_max")"
    generate_samples "$condition" "$network" "$main_steps" "$info"
    if [[ "$SAMPLES_ONLY" == "1" ]]; then
        log "[$condition] SAMPLES_ONLY=1; skipping FID generation and evaluation"
        return 0
    fi

    pool="$IMAGE_ROOT/$condition/pool-seeds-0-49999"
    run_log="$LOG_DIR/$condition-fid${FID_STAGE}-seeds-$(tokenize "$FID_SEEDS").log"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition checkpoint=$checkpoint method=$method main_steps=$main_steps K=$restart_k NFE=$nfe group=$comparison_group]"
        log "PLAN [$condition] restart_info=$info"
        log "PLAN [$condition] add seeds $DELTA_SEEDS to $pool, then calculate FID-$FID_STAGE using seeds $FID_SEEDS"
        return 0
    fi

    if result_done "$condition" "$FID_STAGE" "$FID_SEEDS"; then
        log "[$condition] FID-$FID_STAGE already recorded; skipping"
        return 0
    fi
    require_previous_stage "$condition"

    mkdir -p "$pool"
    png_count="$(count_pngs "$pool")"
    if (( png_count > FID_STAGE )); then
        log "[$condition] pool has $png_count images, exceeding requested stage $FID_STAGE"
        log "[$condition] refusing to let fid.py select a random subset"
        return 1
    fi
    if (( png_count < PREVIOUS_STAGE )); then
        log "[$condition] pool has only $png_count images, below prerequisite stage $PREVIOUS_STAGE"
        return 1
    fi
    if (( png_count < FID_STAGE )); then
        log "[$condition] extending pool from $png_count to $FID_STAGE images with seeds $DELTA_SEEDS"
        generate_start="$SECONDS"
        generate_images "$network" "$pool" "$DELTA_SEEDS" "$main_steps" "$info" "$run_log" 1
        generate_sec="$((SECONDS - generate_start))"
    else
        log "[$condition] pool already has $FID_STAGE images; skipping generation"
    fi

    png_count="$(count_pngs "$pool")"
    if (( png_count != FID_STAGE )); then
        log "[$condition] pool has ${png_count}/${FID_STAGE} images after generation"
        return 1
    fi
    "$PYTHON" "$POOL_VALIDATOR" "$pool" --count "$FID_STAGE" | tee -a "$run_log"

    log "[$condition] calculating FID-$FID_STAGE from the complete nested seed set $FID_SEEDS"
    fid_start="$SECONDS"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$pool" \
        --ref="$FID_REF" \
        --num="$FID_STAGE" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"
    fid_sec="$((SECONDS - fid_start))"
    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$condition] could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$condition" "$checkpoint" "$method" "$main_steps" "$restart_steps" "$restart_k" \
        "$t_min" "$t_max" "$nfe" "$comparison_group" "$FID_STAGE" "$FID_SEEDS" "$fid" \
        "$generate_sec" "$fid_sec" "$network" "$pool" "$run_log" "$FID_REF" >> "$SUMMARY"
    log "[$condition] recorded FID-$FID_STAGE=$fid"
}

aggregate_results() {
    if [[ ! -f "$SUMMARY" ]]; then
        log "No result summary yet; skipping aggregation"
        return 0
    fi
    "$PYTHON" experiments/8/1_restart_sampling_checkpoint_sweep/scripts/summarize_results.py \
        --input "$SUMMARY" \
        --output "$IMPROVEMENTS"
    if awk -F '\t' -v num="$FID_STAGE" -v seeds="$FID_SEEDS" \
        'NR > 1 && $5 == num && $6 == seeds && $8 != "NA" && $10 != "NA" { found = 1 } END { exit(found ? 0 : 1) }' \
        "$IMPROVEMENTS"; then
        "$PYTHON" "$EXP_ROOT/scripts/plot_results.py" \
            --input "$IMPROVEMENTS" \
            --output "$RESULTS_DIR/restart_k5_fid${FID_STAGE}_overview" \
            --num-images "$FID_STAGE" \
            --seeds "$FID_SEEDS"
    else
        log "No checkpoint has a complete Restart/baseline triplet for FID-$FID_STAGE; skipping plot"
    fi
}

main() {
    configure_stage
    check_inputs
    log "Experiment 8-6 progressive FID stage $FID_STAGE"
    log "Nested seed set: $FID_SEEDS"
    log "New seeds for this stage: $DELTA_SEEDS"
    log "Previous required stage: $PREVIOUS_STAGE"
    log "Config: $CONFIG"
    log "Filters: conditions=${CONDITION_FILTER:-(all)} checkpoints=${CHECKPOINT_FILTER:-(all)}"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no GPU work will be started"
    fi

    local condition checkpoint main_steps restart_steps restart_k t_min t_max nfe comparison_group network
    local selected=0
    while IFS=$'\t' read -r condition checkpoint main_steps restart_steps restart_k t_min t_max nfe comparison_group network; do
        [[ -z "${condition:-}" || "$condition" =~ ^# ]] && continue
        matches_filter "$condition" "$CONDITION_FILTER" || continue
        matches_filter "$checkpoint" "$CHECKPOINT_FILTER" || continue
        selected=$((selected + 1))
        run_condition "$condition" "$checkpoint" "$main_steps" "$restart_steps" "$restart_k" \
            "$t_min" "$t_max" "$nfe" "$comparison_group" "$network"
        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after $condition"
            break
        fi
    done < "$CONFIG"

    if (( selected == 0 )); then
        log "No conditions matched the filters"
        exit 1
    fi
    if [[ "$PLAN_ONLY" == "1" || "$SAMPLES_ONLY" == "1" ]]; then
        exit 0
    fi

    aggregate_results
    log "Experiment 8-6 FID-$FID_STAGE stage finished"
}

main "$@"
