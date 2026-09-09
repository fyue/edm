#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

export EXPERIMENT_ID="5-12"
export EXP_ROOT="${EXP_ROOT:-experiments/5/12_ab_26nfe_fid50k_3seeds}"
export CONFIG="${CONFIG:-$EXP_ROOT/config/ab_26nfe_fid50k_3seeds.tsv}"
export SAMPLER="ab"
export STEPS="26"
export EXPECTED_NFE="26"
export MASTER_LOG_NAME="ab-26nfe-fid50k-3seeds.log"
export SUMMARY_NAME="ab_26nfe_fid50k_3seeds_summary.tsv"
export STATS_NAME="ab_26nfe_fid50k_3seeds_stats.tsv"

exec scripts/run_matched_ab_coefficients_fid50k_3seeds.sh "$@"
