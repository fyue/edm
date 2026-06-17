#!/usr/bin/env bash
set -euo pipefail

# Experiment 2: conditional generation sanity check.
#
# Purpose:
# - Keep all work files for this check under experiments/2.
# - Verify that the 120k and 200k CIFAR-10 conditional checkpoints can generate
#   class-controlled samples with generate.py.
# - Generate the same seeds for every class so class changes are easy to compare.
# - Write command logs under experiments/2/logs and generated PNGs under
#   experiments/2/generated.
#
# Run from the EDM repository root:
#   bash experiments/2/scripts/generate_conditional_samples.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments/2"
PYTHON="${PYTHON:-/home/kyoukinkin/data/conda-envs/edm/bin/python}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEEDS="${SEEDS:-0-63}"
STEPS="${STEPS:-18}"
BATCH="${BATCH:-64}"
CLASSES="${CLASSES:-0 1 2 3 4 5 6 7 8 9}"

SNAPSHOT_120K="${ROOT_DIR}/training-runs/cifar10-cond-ddpmpp-edm-managed/network-snapshot-120000.pkl"
SNAPSHOT_200K="${ROOT_DIR}/training-runs/cifar10-cond-ddpmpp-edm-managed/network-snapshot-200000.pkl"

mkdir -p "${EXP_DIR}/generated" "${EXP_DIR}/logs" "${EXP_DIR}/results"

run_generate() {
    local label="$1"
    local network="$2"
    local class_idx="$3"
    local outdir="${EXP_DIR}/generated/${label}/class${class_idx}"
    local log="${EXP_DIR}/logs/generate-${label}-class${class_idx}.log"

    echo "[$(date '+%F %T')] ${label} class=${class_idx} seeds=${SEEDS} steps=${STEPS}" | tee "${log}"
    "${PYTHON}" "${ROOT_DIR}/generate.py" \
        --network="${network}" \
        --outdir="${outdir}" \
        --seeds="${SEEDS}" \
        --class="${class_idx}" \
        --steps="${STEPS}" \
        --batch="${BATCH}" 2>&1 | tee -a "${log}"
}

for class_idx in ${CLASSES}; do
    run_generate "120k" "${SNAPSHOT_120K}" "${class_idx}"
    run_generate "200k" "${SNAPSHOT_200K}" "${class_idx}"
done

"${PYTHON}" "${EXP_DIR}/scripts/make_conditional_grid.py"

echo "Done. Summary image:"
echo "${EXP_DIR}/results/conditional_generation_120k_vs_200k.png"
