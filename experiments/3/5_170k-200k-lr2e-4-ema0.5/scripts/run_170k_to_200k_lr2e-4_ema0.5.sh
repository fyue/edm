#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the improved Experiment 3_3 170k checkpoint to 180k.
# The original 200k plan was stopped after the 180k three-seed evaluation.
# A separate script evaluates the saved ~175k snapshot with three seed ranges.
#
# Plan only:
#   PLAN_ONLY=1 experiments/3/5_170k-200k-lr2e-4-ema0.5/scripts/run_170k_to_200k_lr2e-4_ema0.5.sh
#
# Run:
#   tmux new -s edm-exp3-5
#   experiments/3/5_170k-200k-lr2e-4-ema0.5/scripts/run_170k_to_200k_lr2e-4_ema0.5.sh

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/3/5_170k-200k-lr2e-4-ema0.5}"
export RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr2e-4-ema0.5-170k-200k}"
export FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr2e-4-ema0.5-170k-200k}"
export RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
export LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

export BASE_RUN="${BASE_RUN:-experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k}"
export BASE_STATE="${BASE_STATE:-$BASE_RUN/training-state-170000.pt}"
export BASE_SNAPSHOT="${BASE_SNAPSHOT:-$BASE_RUN/network-snapshot-170000.pkl}"

export START_KIMG="${START_KIMG:-170000}"
export TARGET_KIMG="${TARGET_KIMG:-180000}"
export EVAL_KIMGS="${EVAL_KIMGS:-180000}"
export LR="${LR:-0.0002}"
export EMA="${EMA:-0.5}"

export INTERIM_FID_SEEDS_LIST="${INTERIM_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
export FINAL_FID_SEEDS_LIST="${FINAL_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"

exec experiments/3/2_160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh "$@"
