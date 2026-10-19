#!/usr/bin/env bash
# Source this file before scheduling GPU work. Local policy overrides caller values.
EDM_GPU_CONFIG="${EDM_GPU_CONFIG:-$HOME/.config/edm/gpu.env}"
if [[ ! -r "$EDM_GPU_CONFIG" ]]; then
    printf 'Missing local GPU configuration: %s\n' "$EDM_GPU_CONFIG" >&2
    return 1
fi
unset GPU_INDEX EXPECTED_GPU_NAME
source "$EDM_GPU_CONFIG"
if [[ ! "${GPU_INDEX:-}" =~ ^(0|[1-9][0-9]*)$ || -z "${EXPECTED_GPU_NAME:-}" ]]; then
    printf 'Local GPU configuration must define GPU_INDEX and EXPECTED_GPU_NAME.\n' >&2
    return 1
fi
export GPU_INDEX EXPECTED_GPU_NAME
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
