#!/usr/bin/env bash
# CUDA-only, immutable-output SRA P0 launcher.  Deliberately no CPU fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${1:?usage: $0 OUTPUT_DIR}"
exec "${ROOT}/experiments/run_with_torch_cudnn.sh" \
  "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.run_sra_p0_pointmaze_cuda \
  --output-dir "${STAMP}"
