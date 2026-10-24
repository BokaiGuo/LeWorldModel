#!/usr/bin/env bash
# Host-CUDA-only entrypoint for the independent PushT D2P P0 admission.
# Invoke directly on the host: Codex's isolated sandbox does not expose GPU
# device nodes.  This script deliberately has no CPU fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-results/pusht_d2p_p0_preflight_20261024_cuda}"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
if [[ "$ATTEMPTS" -lt 1 ]]; then
  echo "GPU_PREFLIGHT_ATTEMPTS must be positive" >&2
  exit 2
fi

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  if .venv/bin/python -m experiments.gpu_preflight --device cuda; then
    exec .venv/bin/python -m experiments.spectral_reachability.preflight_pusht_d2p_p0 \
      --output "$OUTPUT" \
      --device cuda \
      --calibration-roots 4 \
      --reserved-outcome-roots 24 \
      --horizon 2 \
      --history-len 3 \
      --action-block 5 \
      --probe-dims 8 \
      --fd-pairs 8 \
      --fd-epsilon 0.001
  fi
  echo "CUDA preflight attempt ${attempt}/${ATTEMPTS} failed; no CPU fallback will be used." >&2
done

echo "CUDA preflight failed after ${ATTEMPTS} attempts; PushT D2P P0 was not run." >&2
exit 1
