#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../../.."

export EXPERIMENT_ID="5-11"
export EXP_ROOT="${EXP_ROOT:-experiments/5/11_pndm_no_rk_18nfe_fid50k_3seeds}"
export CONFIG="${CONFIG:-$EXP_ROOT/config/pndm_no_rk_18nfe_fid50k_3seeds.tsv}"
export SAMPLER="pndm-no-rk"
export STEPS="18"
export EXPECTED_NFE="18"
export MASTER_LOG_NAME="pndm-no-rk-18nfe-fid50k-3seeds.log"
export SUMMARY_NAME="pndm_no_rk_18nfe_fid50k_3seeds_summary.tsv"
export STATS_NAME="pndm_no_rk_18nfe_fid50k_3seeds_stats.tsv"

exec scripts/run_matched_ab_coefficients_fid50k_3seeds.sh "$@"
