#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 0-2: train conditional CIFAR-10 EDM on a deterministic 30k
# subset, evaluate FID every 10k kimg, and stop after an FID plateau.

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

EXP_ROOT="${EXP_ROOT:-experiments/0/2_cifar10_cond_edm_30k_subset}"
RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/cifar10-cond-ddpmpp-edm-seed0-subset30k}"
FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/cifar10-cond-ddpmpp-edm-seed0-subset30k}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"

DATASET="${DATASET:-datasets/cifar10-32x32.zip}"
DATASET_MAX_SIZE="${DATASET_MAX_SIZE:-30000}"
DATASET_RANDOM_SEED="${DATASET_RANDOM_SEED:-0}"
MAX_KIMG="${MAX_KIMG:-60000}"
CHECKPOINT_INTERVAL_KIMG=2500
FID_INTERVAL_KIMG=10000
TICK_KIMG="${TICK_KIMG:-50}"
BATCH_GPU="${BATCH_GPU:-128}"
SEED="${SEED:-0}"

FID_SEEDS=0-49999
FID_NUM_IMAGES=50000
FID_STEPS=18
FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
FID_MIN_DELTA="${FID_MIN_DELTA:-0.1}"
FID_PATIENCE="${FID_PATIENCE:-2}"
EARLY_STOP="${EARLY_STOP:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/train-and-fid.log"
SUMMARY="$RESULTS_DIR/fid_summary.tsv"
STATUS="$RESULTS_DIR/run_status.tsv"
PLATEAU_SCRIPT="$EXP_ROOT/scripts/evaluate_plateau.py"
IMAGE_POOL_SCRIPT="$EXP_ROOT/scripts/inspect_image_pool.py"

log() {
    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf '[PLAN] %s\n' "$*"
    else
        printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
    fi
}

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

is_non_negative_decimal() {
    awk -v value="$1" 'BEGIN { exit(value ~ /^[0-9]+([.][0-9]+)?$/ ? 0 : 1) }'
}

state_kimg() {
    basename "$1" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/' | sed 's/^0*//; s/^$/0/'
}

latest_state() {
    find "$RUN_DIR" -maxdepth 1 -type f -name 'training-state-*.pt' 2>/dev/null | sort | tail -n 1
}

snapshot_path() {
    printf '%s/network-snapshot-%06d.pkl' "$RUN_DIR" "$1"
}

state_path() {
    printf '%s/training-state-%06d.pt' "$RUN_DIR" "$1"
}

ensure_summary() {
    local expected_header=$'kimg\tseeds\tnum_images\tfid\tsnapshot\timages'
    if [[ ! -f "$SUMMARY" ]]; then
        printf '%s\n' "$expected_header" > "$SUMMARY"
        return
    fi
    local actual_header
    actual_header="$(sed -n '1p' "$SUMMARY")"
    [[ "$actual_header" == "$expected_header" ]] || fail "Unexpected summary header in $SUMMARY"
}

fid_already_done() {
    local target="$1"
    awk -F '\t' -v target="$target" '
        NR > 1 && $1 == target { count += 1 }
        END { exit(count == 1 ? 0 : 1) }
    ' "$SUMMARY"
}

validate_summary_config() {
    awk -F '\t' -v seeds="$FID_SEEDS" -v num_images="$FID_NUM_IMAGES" \
        -v interval="$FID_INTERVAL_KIMG" -v maximum="$MAX_KIMG" '
        NR > 1 && ($2 != seeds || $3 != num_images || $1 <= 0 || $1 % interval != 0 || $1 > maximum) {
            exit 1
        }
    ' "$SUMMARY" || fail "FID summary contains rows incompatible with the current experiment: $SUMMARY"
}

validate_existing_run() {
    local artifact artifact_kimg state current=0 target snapshot
    while IFS= read -r artifact; do
        artifact_kimg="$(basename "$artifact" | sed -E 's/^(network-snapshot-|training-state-)([0-9]+)\.(pkl|pt)$/\2/' | sed 's/^0*//; s/^$/0/')"
        [[ "$artifact_kimg" =~ ^[0-9]+$ ]] || fail "Unexpected checkpoint filename: $artifact"
        (( artifact_kimg > 0 && artifact_kimg % CHECKPOINT_INTERVAL_KIMG == 0 )) || \
            fail "Unexpected checkpoint label: $artifact"
        (( artifact_kimg <= MAX_KIMG )) || fail "Checkpoint ${artifact_kimg} exceeds MAX_KIMG=${MAX_KIMG}"
        [[ -s "$(snapshot_path "$artifact_kimg")" ]] || \
            fail "Checkpoint ${artifact_kimg} has no paired snapshot"
        [[ -s "$(state_path "$artifact_kimg")" ]] || \
            fail "Checkpoint ${artifact_kimg} has no paired state"
    done < <(find "$RUN_DIR" -maxdepth 1 -type f \
        \( -name 'network-snapshot-*.pkl' -o -name 'training-state-*.pt' \) 2>/dev/null | sort -u)

    state="$(latest_state || true)"
    if [[ -z "$state" ]]; then
        if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            fail "Run directory is non-empty but has no resumable state: $RUN_DIR"
        fi
        printf '0\n'
        return
    fi

    current="$(state_kimg "$state")"
    (( current % CHECKPOINT_INTERVAL_KIMG == 0 )) || fail "Unexpected state checkpoint: $state"
    (( current <= MAX_KIMG )) || fail "Latest state ${current} exceeds MAX_KIMG=${MAX_KIMG}"
    # Older runs only saved 10k checkpoints; do not require retroactive 2.5k files.
    for ((target = FID_INTERVAL_KIMG; target <= current; target += FID_INTERVAL_KIMG)); do
        snapshot="$(snapshot_path "$target")"
        [[ -s "$snapshot" ]] || fail "Missing snapshot paired with checkpoint ${target}: $snapshot"
        [[ -s "$(state_path "$target")" ]] || fail "Missing state checkpoint: $(state_path "$target")"
    done

    local options="$RUN_DIR/training_options.json"
    [[ -s "$options" ]] || fail "Missing training options for existing run: $options"
    "$PYTHON" - "$options" "$DATASET_MAX_SIZE" "$DATASET_RANDOM_SEED" "$SEED" <<'PY' || \
        fail "Refusing to resume with incompatible options: $options"
import json
import sys

path, max_size, subset_seed, train_seed = sys.argv[1:]
with open(path, 'r', encoding='utf-8') as handle:
    options = json.load(handle)
dataset = options.get('dataset_kwargs', {})
checks = {
    'dataset max_size': (dataset.get('max_size'), int(max_size)),
    'dataset random_seed': (dataset.get('random_seed'), int(subset_seed)),
    'conditional labels': (dataset.get('use_labels'), True),
    'training seed': (options.get('seed'), int(train_seed)),
    'batch size': (options.get('batch_size'), 512),
}
errors = [f'{name}: found {actual!r}, expected {expected!r}'
          for name, (actual, expected) in checks.items() if actual != expected]
if errors:
    raise SystemExit('Incompatible existing run:\n  ' + '\n  '.join(errors))
PY
    printf '%s\n' "$current"
}

validate_config() {
    [[ "$MAX_KIMG" =~ ^[1-9][0-9]*$ ]] || fail "MAX_KIMG must be a positive integer: $MAX_KIMG"
    (( MAX_KIMG % FID_INTERVAL_KIMG == 0 )) || \
        fail "MAX_KIMG must be a multiple of ${FID_INTERVAL_KIMG}: $MAX_KIMG"
    [[ "$DATASET_MAX_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "DATASET_MAX_SIZE must be positive: $DATASET_MAX_SIZE"
    [[ "$DATASET_RANDOM_SEED" =~ ^[0-9]+$ ]] || fail "DATASET_RANDOM_SEED must be non-negative: $DATASET_RANDOM_SEED"
    [[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be non-negative: $SEED"
    [[ "$BATCH_GPU" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_GPU must be positive: $BATCH_GPU"
    (( 512 % BATCH_GPU == 0 )) || fail "BATCH_GPU must divide total batch 512: $BATCH_GPU"
    [[ "$TICK_KIMG" =~ ^[1-9][0-9]*$ ]] || fail "TICK_KIMG must be positive: $TICK_KIMG"
    [[ "$FID_PATIENCE" =~ ^[1-9][0-9]*$ ]] || fail "FID_PATIENCE must be positive: $FID_PATIENCE"
    is_non_negative_decimal "$FID_MIN_DELTA" || fail "FID_MIN_DELTA must be a non-negative decimal: $FID_MIN_DELTA"
    [[ "$EARLY_STOP" == "0" || "$EARLY_STOP" == "1" ]] || fail "EARLY_STOP must be 0 or 1: $EARLY_STOP"
    [[ "$PLAN_ONLY" == "0" || "$PLAN_ONLY" == "1" ]] || fail "PLAN_ONLY must be 0 or 1: $PLAN_ONLY"
    [[ -f "$DATASET" ]] || fail "Missing dataset: $DATASET"
    [[ -f "$PLATEAU_SCRIPT" ]] || fail "Missing plateau evaluator: $PLATEAU_SCRIPT"
    [[ -f "$IMAGE_POOL_SCRIPT" ]] || fail "Missing image pool inspector: $IMAGE_POOL_SCRIPT"
    [[ -x "$PYTHON" ]] || fail "Missing executable Python: $PYTHON"
    [[ -x "$TORCHRUN" ]] || fail "Missing executable torchrun: $TORCHRUN"
    if [[ "$FID_REF" != http://* && "$FID_REF" != https://* ]]; then
        [[ -f "$FID_REF" ]] || fail "Missing FID reference: $FID_REF"
    fi
}

duration_for_kimg() {
    awk -v kimg="$1" 'BEGIN { printf "%.6g", kimg / 1000 }'
}

print_command() {
    printf '[PLAN] command:'
    printf ' %q' "$@"
    printf '\n'
}

train_command() {
    local target="$1"
    local resume_state="${2:-}"
    local resume_args=()
    local resume_kimg=0 checkpoint checkpoint_csv=""
    if [[ -n "$resume_state" ]]; then
        resume_args+=(--resume="$resume_state")
        resume_kimg="$(state_kimg "$resume_state")"
    fi
    for ((checkpoint = (resume_kimg / CHECKPOINT_INTERVAL_KIMG + 1) * CHECKPOINT_INTERVAL_KIMG;
          checkpoint <= target; checkpoint += CHECKPOINT_INTERVAL_KIMG)); do
        checkpoint_csv+="${checkpoint_csv:+,}${checkpoint}"
    done
    TRAIN_COMMAND=(
        "$TORCHRUN" --standalone --nproc_per_node=1 train.py
        --outdir="$RUN_DIR" --nosubdir
        --data="$DATASET" --dataset-max-size="$DATASET_MAX_SIZE"
        --dataset-random-seed="$DATASET_RANDOM_SEED"
        --cond=1 --arch=ddpmpp --precond=edm
        --duration="$(duration_for_kimg "$target")"
        --batch=512 --batch-gpu="$BATCH_GPU"
        --lr=0.001 --ema=0.5 --dropout=0.13 --augment=0.12
        --fp16=0 --seed="$SEED"
        --tick="$TICK_KIMG" --snap=0 --dump=0
        --checkpoint-kimgs="$checkpoint_csv"
        "${resume_args[@]}"
    )
}

run_train_to() {
    local target="$1"
    local resume_state=""
    local checkpoint resume_kimg=0
    resume_state="$(latest_state || true)"
    train_command "$target" "$resume_state"
    if [[ -n "$resume_state" ]]; then
        resume_kimg="$(state_kimg "$resume_state")"
        log "Resuming from $resume_state toward ${target} kimg"
    else
        log "Starting fresh toward ${target} kimg"
    fi
    "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$MASTER_LOG"
    for ((checkpoint = resume_kimg + CHECKPOINT_INTERVAL_KIMG; checkpoint <= target;
          checkpoint += CHECKPOINT_INTERVAL_KIMG)); do
        [[ -s "$(snapshot_path "$checkpoint")" ]] || fail "Missing snapshot after training: $(snapshot_path "$checkpoint")"
        [[ -s "$(state_path "$checkpoint")" ]] || fail "Missing state after training: $(state_path "$checkpoint")"
    done
}

run_fid_for() {
    local target="$1"
    local snapshot images fid_log actual_count missing_count missing_seeds complete fid
    snapshot="$(snapshot_path "$target")"
    images="$FID_ROOT/images-${target}-${FID_SEEDS}"
    fid_log="$LOG_DIR/fid-${target}-${FID_SEEDS}.log"

    if fid_already_done "$target"; then
        log "FID already recorded for ${target} kimg; skipping"
        return
    fi

    IFS=$'\t' read -r actual_count missing_count missing_seeds complete \
        <<< "$("$PYTHON" "$IMAGE_POOL_SCRIPT" --pool="$images" --count="$FID_NUM_IMAGES" --format=tsv)"
    if [[ "$complete" == "1" ]]; then
        log "Reusing ${actual_count} complete FID images for ${target} kimg"
    else
        if (( actual_count > 0 )); then
            log "Found ${actual_count}/${FID_NUM_IMAGES} valid images; generating ${missing_count} missing seeds"
        else
            log "Generating ${FID_NUM_IMAGES} FID images for ${target} kimg"
        fi
        "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --outdir="$images" --seeds="$missing_seeds" --subdirs \
            --sampler=edm --steps="$FID_STEPS" --S_churn=0 \
            --network="$snapshot" 2>&1 | tee -a "$MASTER_LOG"
        IFS=$'\t' read -r actual_count missing_count missing_seeds complete \
            <<< "$("$PYTHON" "$IMAGE_POOL_SCRIPT" --pool="$images" --count="$FID_NUM_IMAGES" --format=tsv)"
        [[ "$complete" == "1" ]] || fail \
            "Expected ${FID_NUM_IMAGES} generated images, found ${actual_count}; ${missing_count} still missing"
    fi

    log "Calculating FID50k for ${target} kimg"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" --ref="$FID_REF" --num="$FID_NUM_IMAGES" \
        2>&1 | tee "$fid_log" | tee -a "$MASTER_LOG"
    fid="$(awk '/^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    [[ -n "$fid" ]] || fail "Could not parse FID from $fid_log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$target" "$FID_SEEDS" "$FID_NUM_IMAGES" "$fid" "$snapshot" "$images" >> "$SUMMARY"
    log "Recorded FID ${fid} for ${target} kimg"
}

plateau_status() {
    local target="$1"
    "$PYTHON" "$PLATEAU_SCRIPT" \
        --summary="$SUMMARY" --min-delta="$FID_MIN_DELTA" \
        --patience="$FID_PATIENCE" --through-kimg="$target" --format=tsv
}

write_status() {
    local outcome="$1"
    local current="$2"
    local best_kimg="$3"
    local best_fid="$4"
    local stale_count="$5"
    local evaluations="$6"
    local temp="$STATUS.tmp"
    printf 'status\tcurrent_kimg\tbest_kimg\tbest_fid\tstale_count\tevaluations\tmin_delta\tpatience\tmax_kimg\n' > "$temp"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$outcome" "$current" "$best_kimg" "$best_fid" "$stale_count" "$evaluations" \
        "$FID_MIN_DELTA" "$FID_PATIENCE" "$MAX_KIMG" >> "$temp"
    mv "$temp" "$STATUS"
}

show_plan() {
    local target resume=""
    log "Dataset subset: ${DATASET_MAX_SIZE} images selected with random seed ${DATASET_RANDOM_SEED}"
    log "Checkpoint interval: ${CHECKPOINT_INTERVAL_KIMG} kimg; FID interval: ${FID_INTERVAL_KIMG} kimg; maximum: ${MAX_KIMG} kimg"
    log "FID: ${FID_NUM_IMAGES} images, seeds ${FID_SEEDS}, EDM ${FID_STEPS} steps, full CIFAR-10 reference ${FID_REF}"
    log "Early stopping: enabled=${EARLY_STOP}, min_delta=${FID_MIN_DELTA}, patience=${FID_PATIENCE}"
    for ((target = FID_INTERVAL_KIMG; target <= MAX_KIMG; target += FID_INTERVAL_KIMG)); do
        train_command "$target" "$resume"
        print_command "${TRAIN_COMMAND[@]}"
        print_command "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
            --outdir="$FID_ROOT/images-${target}-${FID_SEEDS}" --seeds="$FID_SEEDS" --subdirs \
            --sampler=edm --steps="$FID_STEPS" --S_churn=0 --network="$(snapshot_path "$target")"
        print_command "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
            --images="$FID_ROOT/images-${target}-${FID_SEEDS}" --ref="$FID_REF" --num="$FID_NUM_IMAGES"
        resume="$(state_path "$target")"
    done
}

main() {
    validate_config
    if [[ "$PLAN_ONLY" == "1" ]]; then
        show_plan
        return
    fi

    source scripts/load_gpu_config.sh

    mkdir -p "$RUN_DIR" "$FID_ROOT" "$LOG_DIR" "$RESULTS_DIR"
    ensure_summary
    validate_summary_config
    "$PYTHON" "$PLATEAU_SCRIPT" --summary="$SUMMARY" \
        --min-delta="$FID_MIN_DELTA" --patience="$FID_PATIENCE" >/dev/null
    local current
    current="$(validate_existing_run)"
    log "Dataset: $DATASET; subset: ${DATASET_MAX_SIZE}; subset seed: ${DATASET_RANDOM_SEED}"
    log "Run directory: $RUN_DIR; current: ${current} kimg; maximum: ${MAX_KIMG} kimg"
    log "FID50k seeds: $FID_SEEDS; early stop: $EARLY_STOP; min delta: $FID_MIN_DELTA; patience: $FID_PATIENCE"

    local target decision best_kimg best_fid stale_count stop_kimg evaluations
    for ((target = FID_INTERVAL_KIMG; target <= MAX_KIMG; target += FID_INTERVAL_KIMG)); do
        if (( current < target )); then
            run_train_to "$target"
            current="$target"
        else
            log "Checkpoint ${target} kimg already exists; skipping training"
        fi
        [[ -s "$(snapshot_path "$target")" ]] || fail "Missing snapshot: $(snapshot_path "$target")"
        [[ -s "$(state_path "$target")" ]] || fail "Missing state: $(state_path "$target")"

        run_fid_for "$target"
        IFS=$'\t' read -r decision best_kimg best_fid stale_count stop_kimg evaluations \
            <<< "$(plateau_status "$target")"
        log "FID plateau status: best=${best_fid} at ${best_kimg} kimg; stale=${stale_count}/${FID_PATIENCE}"
        write_status running "$target" "$best_kimg" "$best_fid" "$stale_count" "$evaluations"

        if [[ "$EARLY_STOP" == "1" && "$decision" == "1" ]]; then
            write_status early_stop "$stop_kimg" "$best_kimg" "$best_fid" "$stale_count" "$evaluations"
            log "Early stopping at ${stop_kimg} kimg: best FID ${best_fid} at ${best_kimg} kimg; no improvement of ${FID_MIN_DELTA} for ${stale_count} evaluations"
            return
        fi
    done

    decision=0
    IFS=$'\t' read -r decision best_kimg best_fid stale_count stop_kimg evaluations \
        <<< "$(plateau_status "$MAX_KIMG")"
    write_status max_kimg "$MAX_KIMG" "$best_kimg" "$best_fid" "$stale_count" "$evaluations"
    log "Reached MAX_KIMG=${MAX_KIMG}; best FID ${best_fid} at ${best_kimg} kimg"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
