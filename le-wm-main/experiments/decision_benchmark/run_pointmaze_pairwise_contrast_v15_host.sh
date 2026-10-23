#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"
for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
  [[ -c "${node}" ]] || { echo "host CUDA device node is unavailable: ${node}" >&2; exit 3; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable; refusing CPU fallback" >&2; exit 4; }
[[ -f "${ROOT}/results/pointmaze_pairwise_contrast_v15_radial_velocity_alias_static_preflight_20261023.json" ]] || {
  echo "missing V15 passed static preflight ledger" >&2; exit 6;
}
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_pairwise_contrast_v15_cuda
