# Source from the Experiment 8-4 runners after setting EXP_ROOT and PYTHON.
# An explicit CONFIG preserves the legacy fixed-config workflow.

if [[ -z "${CONFIG:-}" ]]; then
    if [[ -n "${CHECKPOINTS:-}" && -n "${CHECKPOINT_FILTER:-}" && "$CHECKPOINTS" != "$CHECKPOINT_FILTER" ]]; then
        printf 'Set either CHECKPOINTS or CHECKPOINT_FILTER, not conflicting values.\n' >&2
        exit 1
    fi
    checkpoint_selection="${CHECKPOINTS:-${CHECKPOINT_FILTER:-all}}"
    checkpoint_args=(--template "$EXP_ROOT/config/restart_k0_8_fid5k.tsv" --checkpoints "$checkpoint_selection")
    if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
        checkpoint_args+=(--checkpoint-dir "$CHECKPOINT_DIR")
    fi
    if [[ "${CHECKPOINT_PROTOCOL:-}" == "fid25k" ]]; then
        checkpoint_args+=(--source-summary "$EXP_ROOT/results/restart_fid_summary.tsv")
    fi
    if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
        checkpoint_config=$(mktemp /tmp/edm-checkpoints.XXXXXX.tsv)
        trap 'unlink "$checkpoint_config"' EXIT
    else
        mkdir -p "${LOG_DIR:-$EXP_ROOT/logs}"
        checkpoint_config=$(mktemp "${LOG_DIR:-$EXP_ROOT/logs}/checkpoint-config.XXXXXX.tsv")
    fi
    "$PYTHON" "$EXP_ROOT/scripts/build_checkpoint_config.py" "${checkpoint_args[@]}" > "$checkpoint_config"
    export CONFIG="$checkpoint_config"
    # Ranges/all have already been expanded into the config.
    export CHECKPOINT_FILTER=""
elif [[ -n "${CHECKPOINTS:-}" ]]; then
    printf 'CHECKPOINTS requires automatic config; omit CONFIG or use CHECKPOINT_FILTER with a fixed CONFIG.\n' >&2
    exit 1
fi
