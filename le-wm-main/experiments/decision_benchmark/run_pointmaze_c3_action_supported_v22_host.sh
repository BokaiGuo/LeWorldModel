#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do [[ -c "${node}" ]] || { echo "missing host CUDA node: ${node}" >&2; exit 3; }; done
"${ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${ROOT}/.venv/bin/python" -m experiments.decision_benchmark.run_pointmaze_c3_action_supported_v22_cuda
