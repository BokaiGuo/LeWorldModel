#!/usr/bin/env bash
# Host-only launcher for the fresh V9 RGB-geometry counterfactual confirmation.
# This must run in the host environment that owns /dev/nvidia*, never inside
# the Codex Desktop sandbox.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"

for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
  if [[ ! -c "${node}" ]]; then
    echo "host CUDA device node is unavailable: ${node}" >&2
    exit 3
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing CPU fallback" >&2
  exit 4
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "missing project Python: ${PYTHON}" >&2
  exit 5
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v9_rgb_geometry_cuda
