#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
OUTPUT="${P3_OUTPUT:-${ROOT}/results/spectral_repair_synthetic_p3_20261023_cuda}"
SEED="${P3_SEED:-202610231}"
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite P3 synthetic evidence" >&2; exit 3; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "P3_SEED must be a non-negative integer" >&2; exit 5; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight --device cuda
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.decision_benchmark.run_spectral_repair_synthetic_p3_cuda --output "${OUTPUT}" --device cuda --seed "${SEED}"
