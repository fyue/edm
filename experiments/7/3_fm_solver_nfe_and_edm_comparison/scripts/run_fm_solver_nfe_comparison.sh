#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 7-3:
# 20k FM checkpointのEuler/Heunを同一NFEで比較し、同じ20kのEDMとも
# 18 NFE (Euler) / 35 NFE (Heun) で比較する評価専用runner。

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
EXP_ROOT="${EXP_ROOT:-experiments/7/3_fm_solver_nfe_and_edm_comparison}"
CONFIG="${CONFIG:-$EXP_ROOT/config/conditions.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
FID_SEEDS="${FID_SEEDS:-0-4999}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-0-63}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
CONDITION_FILTER="${CONDITION_FILTER:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fm-solver-nfe-comparison.log"
SUMMARY="$RESULTS_DIR/fm_solver_nfe_comparison.tsv"
STOP_FILE="$EXP_ROOT/STOP_AFTER_CURRENT_CONDITION"

mkdir -p "$IMAGE_ROOT" "$RESULTS_DIR" "$LOG_DIR"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$MASTER_LOG"
}

is_url() {
    [[ "$1" =~ ^https?:// ]]
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
    local condition="$1"
    local wanted
    if [[ -z "$CONDITION_FILTER" ]]; then
        return 0
    fi
    for wanted in $CONDITION_FILTER; do
        if [[ "$wanted" == "$condition" ]]; then
            return 0
        fi
    done
    return 1
}

ensure_summary() {
    if [[ ! -f "$SUMMARY" ]]; then
        printf 'condition\tmodel\tsampler\tsolver\tsteps\tnfe\tcomparison_group\tnum_images\tseeds\tfid\tgenerate_sec\tfid_sec\timage_source\tnetwork\timages\tlog\tfid_ref\n' > "$SUMMARY"
    fi
}

condition_done() {
    local condition="$1"
    ensure_summary
    awk -F '\t' -v condition="$condition" -v num="$NUM_IMAGES" \
        -v seeds="$FID_SEEDS" -v ref="$FID_REF" \
        'NR > 1 && $1 == condition && $8 == num && $9 == seeds && $17 == ref { found = 1 } END { exit(found ? 0 : 1) }' \
        "$SUMMARY"
}

expected_nfe() {
    local model="$1"
    local sampler="$2"
    local steps="$3"
    local solver="$4"

    if [[ "$model" == "fm" && "$sampler" == "fm-euler" && "$solver" == "-" ]]; then
        printf '%s' "$steps"
    elif [[ "$model" == "fm" && "$sampler" == "fm-heun" && "$solver" == "-" ]]; then
        printf '%s' "$((2 * steps - 1))"
    elif [[ "$model" == "edm" && "$sampler" == "edm" && "$solver" == "euler" ]]; then
        printf '%s' "$steps"
    elif [[ "$model" == "edm" && "$sampler" == "edm" && "$solver" == "-" ]]; then
        # generate.pyの標準EDM samplerは最後だけEulerのHeun法。
        printf '%s' "$((2 * steps - 1))"
    else
        printf 'invalid'
    fi
}

validate_condition() {
    local condition="$1"
    local model="$2"
    local sampler="$3"
    local steps="$4"
    local solver="$5"
    local nfe="$6"
    local network="$7"
    local source_images="$8"
    local calculated

    if [[ ! "$steps" =~ ^[1-9][0-9]*$ || ! "$nfe" =~ ^[1-9][0-9]*$ ]]; then
        log "[$condition] steps and nfe must be positive integers: steps=$steps nfe=$nfe"
        return 1
    fi
    calculated="$(expected_nfe "$model" "$sampler" "$steps" "$solver")"
    if [[ "$calculated" == "invalid" || "$calculated" != "$nfe" ]]; then
        log "[$condition] invalid sampler/NFE combination: model=$model sampler=$sampler solver=$solver steps=$steps configured_nfe=$nfe expected=$calculated"
        return 1
    fi
    if [[ ! -f "$network" ]] && ! is_url "$network"; then
        log "[$condition] missing checkpoint: $network"
        return 1
    fi
    if [[ "$source_images" != "-" ]]; then
        local source_count
        source_count="$(count_pngs "$source_images")"
        if (( source_count < NUM_IMAGES )); then
            log "[$condition] source image directory has ${source_count}/${NUM_IMAGES} PNGs: $source_images"
            return 1
        fi
    fi
}

check_inputs() {
    local missing=0
    if [[ ! -f "$CONFIG" ]]; then
        log "Missing config: $CONFIG"
        missing=1
    fi
    if [[ ! -f "$FID_REF" ]] && ! is_url "$FID_REF"; then
        log "Missing FID reference: $FID_REF"
        missing=1
    fi
    if [[ ! "$NUM_IMAGES" =~ ^[0-9]+$ ]] || (( NUM_IMAGES < 2 )); then
        log "NUM_IMAGES must be an integer >= 2: $NUM_IMAGES"
        missing=1
    fi
    if [[ "$PLAN_ONLY" != "1" ]] && ! command -v "$TORCHRUN" >/dev/null 2>&1; then
        log "Missing executable torchrun: $TORCHRUN"
        missing=1
    fi
    if (( missing != 0 )); then
        exit 1
    fi
}

generate_images() {
    local network="$1"
    local outdir="$2"
    local seeds="$3"
    local sampler="$4"
    local steps="$5"
    local solver="$6"
    local run_log="$7"
    local subdirs="$8"
    local extra_args=()

    if [[ "$solver" != "-" ]]; then
        extra_args+=(--solver="$solver")
    fi
    if [[ "$subdirs" == "1" ]]; then
        extra_args+=(--subdirs)
    fi

    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --network="$network" \
        --outdir="$outdir" \
        --seeds="$seeds" \
        --batch="$GEN_BATCH" \
        --sampler="$sampler" \
        --steps="$steps" \
        "${extra_args[@]}" 2>&1 | tee -a "$run_log"
}

generate_samples() {
    local condition="$1"
    local network="$2"
    local sampler="$3"
    local steps="$4"
    local solver="$5"
    local sample_dir="$IMAGE_ROOT/$condition/samples-seeds-$(tokenize "$SAMPLE_SEEDS")"
    local marker="$sample_dir/.complete"
    local run_log="$LOG_DIR/$condition-samples.log"

    if [[ "$SAMPLE_SEEDS" == "-" ]]; then
        return 0
    fi
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition] sample seeds $SAMPLE_SEEDS -> $sample_dir"
        return 0
    fi
    if [[ -f "$marker" ]]; then
        log "[$condition] fixed-seed samples already complete; skipping"
        return 0
    fi

    log "[$condition] generating fixed-seed samples $SAMPLE_SEEDS"
    mkdir -p "$sample_dir"
    generate_images "$network" "$sample_dir" "$SAMPLE_SEEDS" "$sampler" "$steps" "$solver" "$run_log" 0
    touch "$marker"
}

run_condition() {
    local condition="$1"
    local model="$2"
    local sampler="$3"
    local steps="$4"
    local solver="$5"
    local nfe="$6"
    local comparison_group="$7"
    local network="$8"
    local source_images="$9"
    local images
    local image_source
    local run_log
    local png_count
    local generate_start
    local generate_sec="NA"
    local fid_start
    local fid_sec
    local fid

    validate_condition "$condition" "$model" "$sampler" "$steps" "$solver" "$nfe" "$network" "$source_images"
    generate_samples "$condition" "$network" "$sampler" "$steps" "$solver"

    if condition_done "$condition"; then
        log "[$condition] FID already recorded; skipping"
        return 0
    fi

    run_log="$LOG_DIR/$condition-fid${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS").log"
    if [[ "$source_images" != "-" ]]; then
        images="$source_images"
        image_source="reused"
    else
        images="$IMAGE_ROOT/$condition/fid-n${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS")"
        image_source="generated"
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition model=$model sampler=$sampler solver=$solver steps=$steps NFE=$nfe group=$comparison_group]"
        if [[ "$image_source" == "reused" ]]; then
            log "PLAN [$condition] reuse $NUM_IMAGES images from $images"
        else
            log "PLAN [$condition] generate $NUM_IMAGES images seeds $FID_SEEDS -> $images"
        fi
        log "PLAN [$condition] calculate FID-$NUM_IMAGES against $FID_REF"
        return 0
    fi

    if [[ "$image_source" == "generated" ]]; then
        mkdir -p "$images"
        png_count="$(count_pngs "$images")"
        if (( png_count >= NUM_IMAGES )); then
            log "[$condition] found ${png_count} PNGs; skipping generation"
            image_source="existing"
        else
            log "[$condition] generating $NUM_IMAGES images ($sampler, steps=$steps, NFE=$nfe)"
            generate_start="$SECONDS"
            generate_images "$network" "$images" "$FID_SEEDS" "$sampler" "$steps" "$solver" "$run_log" 1
            generate_sec="$((SECONDS - generate_start))"
        fi
    else
        log "[$condition] reusing $NUM_IMAGES images from $images"
    fi

    png_count="$(count_pngs "$images")"
    if (( png_count < NUM_IMAGES )); then
        log "[$condition] only ${png_count}/${NUM_IMAGES} images are available"
        return 1
    fi

    log "[$condition] calculating FID-$NUM_IMAGES"
    fid_start="$SECONDS"
    "$TORCHRUN" --standalone --nproc_per_node=1 fid.py calc \
        --images="$images" \
        --ref="$FID_REF" \
        --num="$NUM_IMAGES" \
        --batch="$FID_BATCH" 2>&1 | tee -a "$run_log"
    fid_sec="$((SECONDS - fid_start))"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$run_log")"
    if [[ -z "$fid" ]]; then
        log "[$condition] could not parse FID from $run_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$condition" "$model" "$sampler" "$solver" "$steps" "$nfe" "$comparison_group" \
        "$NUM_IMAGES" "$FID_SEEDS" "$fid" "$generate_sec" "$fid_sec" "$image_source" \
        "$network" "$images" "$run_log" "$FID_REF" >> "$SUMMARY"
    log "[$condition] recorded FID-$NUM_IMAGES=$fid (generate_sec=${generate_sec}, fid_sec=${fid_sec})"
}

main() {
    check_inputs
    log "Experiment 7-3: FM solver/NFE and EDM comparison"
    log "Config: $CONFIG"
    log "FID: num=$NUM_IMAGES seeds=$FID_SEEDS ref=$FID_REF"
    log "Samples: seeds=$SAMPLE_SEEDS"
    log "Condition filter: ${CONDITION_FILTER:-(all)}"
    log "Stop after current condition: $STOP_FILE"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN_ONLY=1; no GPU work will be started"
    fi

    local condition
    local model
    local sampler
    local steps
    local solver
    local nfe
    local comparison_group
    local network
    local source_images
    local selected=0

    while IFS=$'\t' read -r condition model sampler steps solver nfe comparison_group network source_images; do
        [[ -z "${condition:-}" || "$condition" =~ ^# ]] && continue
        if ! matches_filter "$condition"; then
            continue
        fi
        selected=$((selected + 1))
        run_condition "$condition" "$model" "$sampler" "$steps" "$solver" "$nfe" \
            "$comparison_group" "$network" "$source_images"

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after $condition"
            exit 0
        fi
    done < "$CONFIG"

    if (( selected == 0 )); then
        log "No conditions matched CONDITION_FILTER=$CONDITION_FILTER"
        exit 1
    fi

    log "Experiment 7-3 complete"
    if [[ "$PLAN_ONLY" != "1" ]]; then
        ensure_summary
        cat "$SUMMARY" | tee -a "$MASTER_LOG"
    fi
}

main "$@"
