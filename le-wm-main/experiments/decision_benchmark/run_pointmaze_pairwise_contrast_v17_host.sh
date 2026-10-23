#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
for n in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do [[ -c "$n" ]] || exit 3; done
"${ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${ROOT}/.venv/bin/python" -m experiments.decision_benchmark.run_pointmaze_pairwise_contrast_v17_cuda
