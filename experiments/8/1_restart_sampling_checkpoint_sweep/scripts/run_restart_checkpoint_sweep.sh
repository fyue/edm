#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 8-1: checkpoint-wise Restart vs ordinary EDM Heun comparison.

cd "$(dirname "$0")/../../../.."
REPO_ROOT="$(pwd -P)"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
EXP_ROOT="${EXP_ROOT:-experiments/8/1_restart_sampling_checkpoint_sweep}"
CONFIG="${CONFIG:-$EXP_ROOT/config/pilot.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
NUM_IMAGES="${NUM_IMAGES:-1024}"
FID_SEEDS="${FID_SEEDS:-0-1023}"
FID_GENERATION_SEEDS="${FID_GENERATION_SEEDS:-$FID_SEEDS}"
REQUIRE_EXACT_IMAGE_COUNT="${REQUIRE_EXACT_IMAGE_COUNT:-0}"
REQUIRE_EXACT_SEED_COVERAGE="${REQUIRE_EXACT_SEED_COVERAGE:-0}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
SAMPLES_ONLY="${SAMPLES_ONLY:-0}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
RHO="${RHO:-7}"
RESTART_NOISE="${RESTART_NOISE:-1}"
CONDITION_FILTER="${CONDITION_FILTER:-}"
CHECKPOINT_FILTER="${CHECKPOINT_FILTER:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-Experiment 8-1: Restart checkpoint sweep}"

MASTER_LOG="${MASTER_LOG:-$LOG_DIR/restart-checkpoint-sweep.log}"
SUMMARY="$RESULTS_DIR/restart_fid_summary.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"
pause_default=0
case "$(realpath -m -- "$EXP_ROOT")" in
    "$REPO_ROOT/experiments/8/4_restart_k_sweep_10k_interval"|"$REPO_ROOT/experiments/8/4_restart_k_sweep_10k_interval/"*) pause_default=1 ;;
esac
PAUSE_ENABLED="${PAUSE_ENABLED:-$pause_default}"
PAUSE_FILE="${PAUSE_FILE:-/data/kyou/STOP_RUNNING_PASCAL}"
PAUSE_POLL_SEC="${PAUSE_POLL_SEC:-1}"
PAUSE_ETA_UPDATE_SEC="${PAUSE_ETA_UPDATE_SEC:-30}"
PAUSE_LOG="$LOG_DIR/pause.log"
GPU_CONFIG_LOADED=0
ETA_HELPER="$REPO_ROOT/experiments/8/4_restart_k_sweep_10k_interval/scripts/pause_eta.py"
ETA_PYTHON="${PYTHON:-python3}"
ETA_PID=""
ETA_STATE="$LOG_DIR/pause-runtime.$$.json"
ETA_HISTORY="$LOG_DIR/pause-timings.jsonl"
ETA_RUN_ID="$$-$(date +%s)"
ETA_CONDITION="-"
ETA_CHECKPOINT="-"
ETA_K="-"
ETA_RUN_LOG=""
ETA_COMPLETE=0

mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

pause_event() {
    log "$*" | tee -a "$PAUSE_LOG"
}

eta_phase() {
    if [[ -z "$ETA_PID" ]]; then
        return 0
    fi
    local extra=()
    if [[ -n "$ETA_RUN_LOG" ]]; then
        extra+=(--log "$ETA_RUN_LOG")
    fi
    if ! "$ETA_PYTHON" "$ETA_HELPER" state --state "$ETA_STATE" --history "$ETA_HISTORY" \
        --phase "$1" --run-id "$ETA_RUN_ID" --condition "$ETA_CONDITION" \
        --checkpoint "$ETA_CHECKPOINT" --k "$ETA_K" --num-images "$NUM_IMAGES" \
        --fid-ref "$FID_REF" "${extra[@]}"; then
        pause_event "STOP ETA warning: could not publish phase $1; condition-boundary pause remains active"
    fi
}

eta_cleanup() {
    local status="$?"
    if [[ -n "$ETA_PID" ]]; then
        if (( status != 0 )); then
            eta_phase error
        elif [[ "$ETA_COMPLETE" == "1" ]]; then
            eta_phase complete
        else
            eta_phase exit
        fi
        # Only the observer is signalled; generation/FID processes are never paused here.
        kill -TERM "$ETA_PID" 2>/dev/null || true
        wait "$ETA_PID" 2>/dev/null || true
    fi
    return "$status"
}

start_eta_monitor() {
    if [[ "$PLAN_ONLY" == "1" || "$PAUSE_ENABLED" != "1" ]]; then
        return 0
    fi
    if ! command -v "$ETA_PYTHON" >/dev/null 2>&1 || [[ ! -f "$ETA_HELPER" ]]; then
        pause_event "STOP ETA warning: Python/helper unavailable; condition-boundary pause remains active"
        return 0
    fi
    "$ETA_PYTHON" "$ETA_HELPER" watch --state "$ETA_STATE" --history "$ETA_HISTORY" \
        --pause-file "$PAUSE_FILE" --summary "$SUMMARY" --parent-pid "$$" \
        --poll-sec "$PAUSE_POLL_SEC" --update-sec "$PAUSE_ETA_UPDATE_SEC" \
        >> "$MASTER_LOG" 2>&1 &
    ETA_PID="$!"
    trap eta_cleanup EXIT
    eta_phase boundary
}

exit_if_stop_requested() {
    if [[ -f "$STOP_FILE" ]]; then
        log "Found stop file; exiting before next condition: $STOP_FILE"
        exit 0
    fi
}

wait_at_condition_boundary() {
    local condition="$1"
    local paused=0
    if [[ "$PLAN_ONLY" == "1" || "$PAUSE_ENABLED" != "1" ]]; then
        return 0
    fi

    # No checks inside generation/FID: the current condition is always saved first.
    while true; do
        exit_if_stop_requested
        if [[ -e "$PAUSE_FILE" ]]; then
            eta_phase waiting
            pause_event "PAUSE before [$condition FID-$NUM_IMAGES]; waiting for removal: $PAUSE_FILE"
            paused=1
            while [[ -e "$PAUSE_FILE" ]]; do
                exit_if_stop_requested
                sleep "$PAUSE_POLL_SEC"
            done
        fi
        exit_if_stop_requested
        if [[ "$GPU_CONFIG_LOADED" == "0" || "$paused" == "1" ]]; then
            source "$REPO_ROOT/scripts/load_gpu_config.sh"
            GPU_CONFIG_LOADED=1
        fi
        if [[ "$paused" == "1" ]]; then
            eta_phase gpu_wait
            pause_event "RESUME checking configured GPU before [$condition FID-$NUM_IMAGES]"
            RUN_SCRIPT=/bin/true EXP_ROOT="$EXP_ROOT" LOG_DIR="$LOG_DIR" \
                SCHEDULE_LABEL="Experiment 8-4 resume before $condition FID-$NUM_IMAGES" \
                "$REPO_ROOT/scripts/wait_for_gpu_and_run.sh"
            source "$REPO_ROOT/scripts/load_gpu_config.sh"
            exit_if_stop_requested
        fi
        # A pause may have been requested while the common GPU waiter was running.
        if [[ -e "$PAUSE_FILE" ]]; then
            continue
        fi
        if [[ "$paused" == "1" ]]; then
            pause_event "RESUME GPU free; continuing [$condition FID-$NUM_IMAGES]"
        fi
        return 0
    done
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

validate_seed_coverage() {
    local images="$1"
    local seeds="$2"
    local seed_start
    local seed_end

    if [[ ! "$seeds" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        log "Exact seed coverage requires one inclusive range, got: $seeds"
        return 1
    fi
    seed_start="${BASH_REMATCH[1]}"
    seed_end="${BASH_REMATCH[2]}"
    if (( seed_end < seed_start )); then
        log "Invalid exact seed range: $seeds"
        return 1
    fi

    if ! find "$images" -type f -name '*.png' -printf '%P\n' | awk \
        -v seed_start="$seed_start" -v seed_end="$seed_end" '
            {
                actual[$0]++
            }
            END {
                failures = 0
                for (seed = seed_start; seed <= seed_end; seed++) {
                    subdir = seed - (seed % 1000)
                    expected = sprintf("%06d/%06d.png", subdir, seed)
                    wanted[expected] = 1
                    if (actual[expected] != 1) {
                        if (failures < 10) {
                            printf "Missing or duplicate expected image: %s (count=%d)\n", expected, actual[expected] > "/dev/stderr"
                        }
                        failures++
                    }
                }
                for (path in actual) {
                    if (!(path in wanted)) {
                        if (failures < 10) {
                            printf "Unexpected image path: %s\n", path > "/dev/stderr"
                        }
                        failures++
                    }
                }
                exit(failures == 0 ? 0 : 1)
            }
        '; then
        log "Image seed coverage does not exactly match $seeds in $images"
        return 1
    fi
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

condition_done() {
    local condition="$1"
    if [[ ! -f "$SUMMARY" ]]; then
        return 1
    fi
    awk -F '\t' -v condition="$condition" -v num="$NUM_IMAGES" \
        -v seeds="$FID_SEEDS" -v ref="$FID_REF" \
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
    if [[ ! "$NUM_IMAGES" =~ ^[0-9]+$ ]] || (( NUM_IMAGES < 2 )); then
        log "NUM_IMAGES must be an integer >= 2: $NUM_IMAGES"
        failed=1
    fi
    if [[ -z "$FID_GENERATION_SEEDS" ]]; then
        log "FID_GENERATION_SEEDS must not be empty"
        failed=1
    fi
    if [[ "$REQUIRE_EXACT_IMAGE_COUNT" != "0" && "$REQUIRE_EXACT_IMAGE_COUNT" != "1" ]]; then
        log "REQUIRE_EXACT_IMAGE_COUNT must be 0 or 1: $REQUIRE_EXACT_IMAGE_COUNT"
        failed=1
    fi
    if [[ "$REQUIRE_EXACT_SEED_COVERAGE" != "0" && "$REQUIRE_EXACT_SEED_COVERAGE" != "1" ]]; then
        log "REQUIRE_EXACT_SEED_COVERAGE must be 0 or 1: $REQUIRE_EXACT_SEED_COVERAGE"
        failed=1
    fi
    if [[ "$PAUSE_ENABLED" != "0" && "$PAUSE_ENABLED" != "1" ]]; then
        log "PAUSE_ENABLED must be 0 or 1: $PAUSE_ENABLED"
        failed=1
    fi
    if [[ "$PAUSE_ENABLED" == "1" ]] && ! awk -v interval="$PAUSE_POLL_SEC" \
        'BEGIN { exit(!(interval ~ /^[0-9]+([.][0-9]+)?$/ && interval > 0 && interval < 1e308)) }'; then
        log "PAUSE_POLL_SEC must be positive and finite: $PAUSE_POLL_SEC"
        failed=1
    fi
    if [[ "$PAUSE_ENABLED" == "1" ]] && ! awk -v interval="$PAUSE_ETA_UPDATE_SEC" \
        'BEGIN { exit(!(interval ~ /^[0-9]+([.][0-9]+)?$/ && interval > 0 && interval < 1e308)) }'; then
        log "PAUSE_ETA_UPDATE_SEC must be positive and finite: $PAUSE_ETA_UPDATE_SEC"
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
    local images
    local run_log
    local png_count
    local generate_start
    local generate_sec="NA"
    local fid_start
    local fid_sec
    local fid
    local samples_pending=0
    local fid_pending=0

    validate_condition "$condition" "$main_steps" "$restart_steps" "$restart_k" "$t_min" "$t_max" "$nfe" "$network"
    if (( restart_k > 0 )); then
        method="restart"
    fi
    info="$(restart_json "$restart_steps" "$restart_k" "$t_min" "$t_max")"
    if [[ "$SAMPLE_SEEDS" != "-" && ! -f "$IMAGE_ROOT/$condition/samples-seeds-$(tokenize "$SAMPLE_SEEDS")/.complete" ]]; then
        samples_pending=1
    fi
    if [[ "$SAMPLES_ONLY" != "1" ]] && ! condition_done "$condition"; then
        fid_pending=1
    fi
    if (( samples_pending || fid_pending )); then
        ETA_CONDITION="$condition"
        ETA_CHECKPOINT="$checkpoint"
        ETA_K="$restart_k"
        ETA_RUN_LOG=""
        eta_phase boundary
        wait_at_condition_boundary "$condition"
    fi
    if (( samples_pending )); then
        eta_phase samples
    fi
    generate_samples "$condition" "$network" "$main_steps" "$info"
    if [[ "$SAMPLES_ONLY" == "1" ]]; then
        log "[$condition] SAMPLES_ONLY=1; skipping FID generation and evaluation"
        return 0
    fi

    if condition_done "$condition"; then
        log "[$condition] FID already recorded; skipping"
        return 0
    fi

    images="$IMAGE_ROOT/$condition/fid-n${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS")"
    run_log="$LOG_DIR/$condition-fid${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS").log"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition checkpoint=$checkpoint method=$method main_steps=$main_steps K=$restart_k NFE=$nfe group=$comparison_group]"
        log "PLAN [$condition] restart_info=$info"
        log "PLAN [$condition] generate seeds $FID_GENERATION_SEEDS into the $NUM_IMAGES-image set"
        log "PLAN [$condition] calculate FID against $FID_REF using seeds $FID_SEEDS"
        return 0
    fi

    mkdir -p "$images"
    png_count="$(count_pngs "$images")"
    if (( png_count >= NUM_IMAGES )); then
        log "[$condition] found ${png_count} PNGs; skipping generation"
    else
        log "[$condition] generating seeds $FID_GENERATION_SEEDS (method=$method, NFE=$nfe)"
        generate_start="$SECONDS"
        ETA_RUN_LOG="$run_log"
        eta_phase generate
        generate_images "$network" "$images" "$FID_GENERATION_SEEDS" "$main_steps" "$info" "$run_log" 1
        generate_sec="$((SECONDS - generate_start))"
    fi

    eta_phase validate
    png_count="$(count_pngs "$images")"
    if [[ "$REQUIRE_EXACT_IMAGE_COUNT" == "1" ]] && (( png_count != NUM_IMAGES )); then
        log "[$condition] found ${png_count} PNGs, but exact count ${NUM_IMAGES} is required"
        return 1
    fi
    if (( png_count < NUM_IMAGES )); then
        log "[$condition] only ${png_count}/${NUM_IMAGES} images are available"
        return 1
    fi
    if [[ "$REQUIRE_EXACT_SEED_COVERAGE" == "1" ]]; then
        log "[$condition] validating exact seed coverage: $FID_SEEDS"
        validate_seed_coverage "$images" "$FID_SEEDS"
    fi

    log "[$condition] calculating FID-$NUM_IMAGES"
    fid_start="$SECONDS"
    ETA_RUN_LOG="$run_log"
    eta_phase fid
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"
    fid_sec="$((SECONDS - fid_start))"
    eta_phase save
    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$condition] could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$condition" "$checkpoint" "$method" "$main_steps" "$restart_steps" "$restart_k" \
        "$t_min" "$t_max" "$nfe" "$comparison_group" "$NUM_IMAGES" "$FID_SEEDS" "$fid" \
        "$generate_sec" "$fid_sec" "$network" "$images" "$run_log" "$FID_REF" >> "$SUMMARY"
    log "[$condition] recorded FID-$NUM_IMAGES=$fid"
    eta_phase saved
}

main() {
    check_inputs
    start_eta_monitor
    log "$EXPERIMENT_LABEL"
    log "Config: $CONFIG"
    log "FID: num=$NUM_IMAGES seeds=$FID_SEEDS ref=$FID_REF"
    log "FID generation seeds: $FID_GENERATION_SEEDS"
    log "Require exact image count: $REQUIRE_EXACT_IMAGE_COUNT"
    log "Require exact seed coverage: $REQUIRE_EXACT_SEED_COVERAGE"
    log "Fixed samples: $SAMPLE_SEEDS"
    log "Samples only: $SAMPLES_ONLY"
    log "Filters: conditions=${CONDITION_FILTER:-(all)} checkpoints=${CHECKPOINT_FILTER:-(all)}"
    if [[ "$PAUSE_ENABLED" == "1" ]]; then
        log "Condition-boundary pause: file=$PAUSE_FILE poll=${PAUSE_POLL_SEC}s"
    fi
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
            exit 0
        fi
    done < "$CONFIG"

    if (( selected == 0 )); then
        log "No conditions matched the filters"
        exit 1
    fi
    log "$EXPERIMENT_LABEL complete"
    ETA_COMPLETE=1
}

main "$@"
