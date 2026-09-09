#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the best Experiment 3 branch from 160k to 170k with lr=0.0002.
# The continuation runner is shared with the existing 160k -> 200k experiment;
# every output path and experiment parameter is overridden below.

cd "$(dirname "$0")/../../../.."

export EXP_ROOT="${EXP_ROOT:-experiments/3/3_160k-170k-lr2e-4-ema0.5}"
export RUN_DIR="${RUN_DIR:-$EXP_ROOT/training-runs/lr2e-4-ema0.5-160k-170k}"
export FID_ROOT="${FID_ROOT:-$EXP_ROOT/fid-evals/lr2e-4-ema0.5-160k-170k}"
export RESULTS_DIR="${RESULTS_DIR:-$EXP_ROOT/results}"
export LOG_DIR="${LOG_DIR:-$EXP_ROOT/logs}"

export BASE_RUN="${BASE_RUN:-experiments/3/1_120k-160k-lr-ema-sweep/training-runs/lr5e-4-ema0.5}"
export START_KIMG="${START_KIMG:-160000}"
export TARGET_KIMG="${TARGET_KIMG:-170000}"
export EVAL_KIMGS="${EVAL_KIMGS:-170000}"
export LR="${LR:-0.0002}"
export EMA="${EMA:-0.5}"

export FINAL_FID_SEEDS_LIST="${FINAL_FID_SEEDS_LIST:-0-49999 50000-99999 100000-149999}"

exec experiments/3/2_160k-200k-lr5e-4-ema0.5/scripts/run_160k_to_200k_lr5e-4_ema0.5.sh "$@"
