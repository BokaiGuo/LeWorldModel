#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
LOG_DIR="${ROOT}/results/pointmaze_c3_stratified_repair_v24_20261023_cuda"
mkdir -p "${LOG_DIR}"
exec >> "${LOG_DIR}/host_runtime.log" 2>&1
for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do [[ -c "${node}" ]] || { echo "missing host CUDA node: ${node}" >&2; exit 3; }; done
"${ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${ROOT}/.venv/bin/python" -m experiments.decision_benchmark.run_pointmaze_c3_stratified_repair_v24_cuda
