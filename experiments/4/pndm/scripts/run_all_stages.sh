#!/usr/bin/env bash
set -Eeuo pipefail

# Run the four Experiment 4 PNDM stages sequentially.
# Each stage is resumable through its own fid_summary.tsv.

cd "$(dirname "$0")/../../../.."

PNDM_ROOT="experiments/4/pndm"
ALL_LOG_DIR="${ALL_LOG_DIR:-$PNDM_ROOT/logs}"
MASTER_LOG="${MASTER_LOG:-$ALL_LOG_DIR/run-all-stages.log}"
PLAN_ONLY="${PLAN_ONLY:-0}"
CONFIRM_STAGE2_CONFIG="${CONFIRM_STAGE2_CONFIG:-0}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2_1="${RUN_STAGE2_1:-1}"
RUN_STAGE2_2="${RUN_STAGE2_2:-1}"
SUMMARIZE_AFTER_RUN="${SUMMARIZE_AFTER_RUN:-1}"

log() {
    local message
    message="$(printf '[%(%Y-%m-%d %H:%M:%S)T] %s' -1 "$*")"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        printf '%s\n' "$message"
    else
        printf '%s\n' "$message" | tee -a "$MASTER_LOG"
    fi
}

run_stage() {
    local label="$1"
    local script="$2"
    log "Starting $label: $script"
    PLAN_ONLY="$PLAN_ONLY" "$script"
    log "Finished $label"
}

main() {
    if [[ "$PLAN_ONLY" != "1" ]]; then
        mkdir -p "$ALL_LOG_DIR"
    fi

    if [[ "$PLAN_ONLY" != "1" && "$CONFIRM_STAGE2_CONFIG" != "1" ]] \
        && { [[ "$RUN_STAGE2_1" == "1" ]] || [[ "$RUN_STAGE2_2" == "1" ]]; }; then
        log "Stage 2 execution is enabled, but CONFIRM_STAGE2_CONFIG is not 1."
        log "Review the Stage 1 results and Stage 2 configs, then rerun with CONFIRM_STAGE2_CONFIG=1."
        return 2
    fi

    log "PNDM sequential run started"
    log "RUN_STAGE0=$RUN_STAGE0 RUN_STAGE1=$RUN_STAGE1 RUN_STAGE2_1=$RUN_STAGE2_1 RUN_STAGE2_2=$RUN_STAGE2_2"

    if [[ "$RUN_STAGE0" == "1" ]]; then
        run_stage "Stage 0 FID-1k" \
            "$PNDM_ROOT/stage0/1_fid1k_churn_sweep/scripts/run.sh"
    fi
    if [[ "$RUN_STAGE1" == "1" ]]; then
        run_stage "Stage 1 FID-5k" \
            "$PNDM_ROOT/stage1/1_fid5k_churn_sweep/scripts/run.sh"
    fi
    if [[ "$RUN_STAGE2_1" == "1" ]]; then
        run_stage "Stage 2-1 FID-50k" \
            "$PNDM_ROOT/stage2/1_fid50k_selected_churn/scripts/run.sh"
    fi
    if [[ "$RUN_STAGE2_2" == "1" ]]; then
        run_stage "Stage 2-2 extra seeds" \
            "$PNDM_ROOT/stage2/2_fid50k_extra_seeds/scripts/run.sh"
    fi

    if [[ "$PLAN_ONLY" != "1" && "$SUMMARIZE_AFTER_RUN" == "1" && "$RUN_STAGE2_2" == "1" ]]; then
        log "Starting Stage 2 three-seed summary"
        "$PNDM_ROOT/stage2/2_fid50k_extra_seeds/scripts/summarize.sh"
        log "Finished Stage 2 three-seed summary"
    fi
    log "PNDM sequential run completed"
}

main "$@"
