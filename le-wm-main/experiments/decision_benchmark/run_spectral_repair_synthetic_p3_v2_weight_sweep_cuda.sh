#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
OUTPUT="${P3V2_SWEEP_OUTPUT:-${ROOT}/results/spectral_repair_synthetic_p3_v2_task_metric_sweep_r2_20261023_cuda}"
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite P3v2 metric-sweep evidence" >&2; exit 3; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight --device cuda
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.decision_benchmark.run_spectral_repair_synthetic_p3_v2_weight_sweep_cuda --output "${OUTPUT}" --device cuda
