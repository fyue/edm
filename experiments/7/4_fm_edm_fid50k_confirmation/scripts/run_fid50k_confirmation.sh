#!/usr/bin/env bash
set -Eeuo pipefail

# Experiment 7-4:
# Experiment 7-3で選んだFM/EDMの代表3条件をFID-50kで再評価する。

cd "$(dirname "$0")/../../../.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCHRUN="${TORCHRUN:-/home/kyoukinkin/data/conda-envs/edm/bin/torchrun}"
EXP_ROOT="${EXP_ROOT:-experiments/7/4_fm_edm_fid50k_confirmation}"
CONFIG="${CONFIG:-$EXP_ROOT/config/conditions.tsv}"
IMAGE_ROOT="${IMAGE_ROOT:-$EXP_ROOT/images}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

FID_REF="${FID_REF:-fid-refs/cifar10-32x32.npz}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
FID_SEEDS="${FID_SEEDS:-0-49999}"
GEN_BATCH="${GEN_BATCH:-64}"
FID_BATCH="${FID_BATCH:-64}"
CONDITION_FILTER="${CONDITION_FILTER:-}"
PLAN_ONLY="${PLAN_ONLY:-0}"

MASTER_LOG="$LOG_DIR/fid50k-confirmation.log"
SUMMARY="$RESULTS_DIR/fid50k_confirmation.tsv"
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
        printf 'condition\tmodel\tsampler\tsolver\tsteps\tnfe\tnum_images\tseeds\tfid_5k\tfid_50k\tgenerate_sec\tfid_sec\timage_source\tnetwork\timages\tgenerate_log\tfid_log\tfid_ref\n' > "$SUMMARY"
    fi
}

condition_done() {
    local condition="$1"
    ensure_summary
    awk -F '\t' -v condition="$condition" -v num="$NUM_IMAGES" \
        -v seeds="$FID_SEEDS" -v ref="$FID_REF" \
        'NR > 1 && $1 == condition && $7 == num && $8 == seeds && $18 == ref { found = 1 } END { exit(found ? 0 : 1) }' \
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
        printf '%s' "$((2 * steps - 1))"
    else
        printf 'invalid'
    fi
}

validate_seed_range() {
    if [[ ! "$FID_SEEDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        log "FID_SEEDS must be one inclusive range such as 0-49999: $FID_SEEDS"
        return 1
    fi
    local first_seed="${BASH_REMATCH[1]}"
    local last_seed="${BASH_REMATCH[2]}"
    local range_size="$((last_seed - first_seed + 1))"
    if (( range_size != NUM_IMAGES )); then
        log "FID_SEEDS contains $range_size seeds but NUM_IMAGES=$NUM_IMAGES: $FID_SEEDS"
        return 1
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
    validate_seed_range
}

generate_images() {
    local network="$1"
    local outdir="$2"
    local sampler="$3"
    local steps="$4"
    local solver="$5"
    local generate_log="$6"
    local extra_args=()

    if [[ "$solver" != "-" ]]; then
        extra_args+=(--solver="$solver")
    fi

    "$TORCHRUN" --standalone --nproc_per_node=1 generate.py \
        --network="$network" \
        --outdir="$outdir" \
        --seeds="$FID_SEEDS" \
        --subdirs \
        --batch="$GEN_BATCH" \
        --sampler="$sampler" \
        --steps="$steps" \
        "${extra_args[@]}" 2>&1 | tee "$generate_log"
}

run_condition() {
    local condition="$1"
    local model="$2"
    local sampler="$3"
    local steps="$4"
    local solver="$5"
    local nfe="$6"
    local fid_5k="$7"
    local network="$8"
    local source_images="$9"
    local images
    local image_source
    local generate_log="$LOG_DIR/$condition-generate.log"
    local fid_log="$LOG_DIR/$condition-fid${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS").log"
    local png_count
    local generate_start
    local generate_sec="NA"
    local fid_start
    local fid_sec
    local fid

    validate_condition "$condition" "$model" "$sampler" "$steps" "$solver" "$nfe" \
        "$network" "$source_images"

    if condition_done "$condition"; then
        log "[$condition] FID-$NUM_IMAGES already recorded; skipping"
        return 0
    fi

    if [[ "$source_images" != "-" ]]; then
        images="$source_images"
        image_source="reused"
    else
        images="$IMAGE_ROOT/$condition/fid-n${NUM_IMAGES}-seeds-$(tokenize "$FID_SEEDS")"
        image_source="generated"
    fi

    if [[ "$PLAN_ONLY" == "1" ]]; then
        log "PLAN [$condition model=$model sampler=$sampler solver=$solver steps=$steps NFE=$nfe FID-5k=$fid_5k]"
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
            if (( png_count > 0 )); then
                log "[$condition] found partial ${png_count}/${NUM_IMAGES} PNGs; regenerating fixed seed range"
            fi
            log "[$condition] generating $NUM_IMAGES images ($sampler, steps=$steps, NFE=$nfe)"
            generate_start="$SECONDS"
            generate_images "$network" "$images" "$sampler" "$steps" "$solver" "$generate_log"
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
        --batch="$FID_BATCH" 2>&1 | tee "$fid_log"
    fid_sec="$((SECONDS - fid_start))"

    fid="$(awk '/^[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$/ { value = $1 } END { print value }' "$fid_log")"
    if [[ -z "$fid" ]]; then
        log "[$condition] could not parse FID from $fid_log"
        return 1
    fi

    ensure_summary
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$condition" "$model" "$sampler" "$solver" "$steps" "$nfe" "$NUM_IMAGES" \
        "$FID_SEEDS" "$fid_5k" "$fid" "$generate_sec" "$fid_sec" "$image_source" \
        "$network" "$images" "$generate_log" "$fid_log" "$FID_REF" >> "$SUMMARY"
    log "[$condition] recorded FID-$NUM_IMAGES=$fid (generate_sec=$generate_sec, fid_sec=$fid_sec)"
}

main() {
    check_inputs
    ensure_summary
    log "Experiment 7-4: FM/EDM FID-50k confirmation"
    log "Config: $CONFIG"
    log "FID: num=$NUM_IMAGES seeds=$FID_SEEDS ref=$FID_REF"
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
    local fid_5k
    local network
    local source_images
    local selected=0

    while IFS=$'\t' read -r condition model sampler steps solver nfe fid_5k network source_images; do
        [[ -z "${condition:-}" || "$condition" =~ ^# ]] && continue
        if ! matches_filter "$condition"; then
            continue
        fi
        selected=$((selected + 1))
        run_condition "$condition" "$model" "$sampler" "$steps" "$solver" "$nfe" \
            "$fid_5k" "$network" "$source_images"

        if [[ -f "$STOP_FILE" ]]; then
            log "Found stop file; exiting after $condition"
            exit 0
        fi
    done < "$CONFIG"

    if (( selected == 0 )); then
        log "No conditions matched CONDITION_FILTER=$CONDITION_FILTER"
        exit 1
    fi

    log "Experiment 7-4 complete"
    if [[ "$PLAN_ONLY" != "1" ]]; then
        cat "$SUMMARY" | tee -a "$MASTER_LOG"
    fi
}

main "$@"
