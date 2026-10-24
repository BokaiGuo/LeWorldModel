#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:?usage: $0 OUTPUT_DIR}"
exec "${ROOT}/experiments/run_with_torch_cudnn.sh" "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.run_sra_p1_pointmaze_cuda --output-dir "${OUTPUT_DIR}"
