#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT}/results/d2p_reacher_p0_20261024_cuda}"
cd "${ROOT}"
exec .venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --device cuda \
  --output "${OUTPUT}"
