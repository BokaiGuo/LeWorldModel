#!/usr/bin/env bash
# Host-only launcher for the V8 CUDA counterfactual-repair confirmation.
#
# Do not invoke this through the Codex desktop sandbox or through
# run_with_torch_cudnn.sh: both can change device visibility/library resolution.
# Run it directly from a host terminal that owns /dev/nvidia*.
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

# Keep this as the final exec: the Python process that passes the CUDA
# preflight owns the complete V8 build, fit, calibration, and score sequence.
exec "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_counterfactual_contrast_v8_robust_cuda
