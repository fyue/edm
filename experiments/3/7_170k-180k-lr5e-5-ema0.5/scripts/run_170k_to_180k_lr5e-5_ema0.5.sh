#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the improved Experiment 3_3 170k checkpoint with lr=0.00005.
# Training stages: 170k -> 175k -> 180k
# FID evaluation: three 50k-seed ranges at 170k, 175k, and 180k.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/3/7_170k-180k-lr5e-5-ema0.5}"
export RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr5e-5-ema0.5-170k-180k}"
export FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr5e-5-ema0.5-170k-180k}"
export RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
export LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

export BASE_RUN="${BASE_RUN:-experiments/3/3_160k-170k-lr2e-4-ema0.5/training-runs/lr2e-4-ema0.5-160k-170k}"
export BASE_STATE="${BASE_STATE:-$BASE_RUN/training-state-170000.pt}"
export BASE_SNAPSHOT="${BASE_SNAPSHOT:-$BASE_RUN/network-snapshot-170000.pkl}"

export START_KIMG="${START_KIMG:-170000}"
export TARGET_KIMG="${TARGET_KIMG:-180000}"
export EVAL_KIMGS="${EVAL_KIMGS:-170000 175000 180000}"
export BASE_EVAL_KIMG="${BASE_EVAL_KIMG:-170000}"
export BASE_EVAL_LR="${BASE_EVAL_LR:-0.0002}"
export BASE_EVAL_EMA="${BASE_EVAL_EMA:-0.5}"
export LR="${LR:-0.00005}"
export EMA="${EMA:-0.5}"

export INTERIM_FID_SEEDS_LIST="${INTERIM_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"
export FINAL_FID_SEEDS_LIST="${FINAL_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"

exec experiments/3/2_160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh "$@"
