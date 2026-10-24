#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
OUTPUT="${ROOT}/results/decision_spectral_synthetic_p2_20261023_cuda"
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite P2 synthetic evidence" >&2; exit 3; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.run_decision_spectral_synthetic_p2_cuda --output "${OUTPUT}" --device cuda
