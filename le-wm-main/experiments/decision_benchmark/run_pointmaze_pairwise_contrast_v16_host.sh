#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"
for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do [[ -c "${node}" ]] || exit 3; done
[[ -f "${ROOT}/results/pointmaze_pairwise_contrast_v16_visible_radial_velocity_alias_static_preflight_20261023.json" ]] || exit 6
"${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_pairwise_contrast_v16_cuda
