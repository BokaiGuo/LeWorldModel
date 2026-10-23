#!/usr/bin/env bash
# GPU-only launcher for the PushT candidate-bank/horizon feasibility gate.
# Pymunk rollouts are CPU physics, but this prospective learned-panel contract
# must never silently be executed as a CPU substitute when CUDA is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${PUSHT_CANDIDATE_HORIZON_PREFLIGHT_OUTPUT:-results/pusht_candidate_horizon_preflight_20261022_cuda}"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
if [[ "$ATTEMPTS" -lt 1 ]]; then
  echo "GPU_PREFLIGHT_ATTEMPTS must be positive" >&2
  exit 2
fi

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  if .venv/bin/python -m experiments.gpu_preflight --device cuda --recurrent-smoke; then
    exec .venv/bin/python -m experiments.preflight_pusht_candidate_horizon \
      --output "$OUTPUT" \
      --device cuda \
      --calibration-roots 16 \
      --outcome-roots 32 \
      --minimum-reachable-roots 1 \
      --minimum-reachable-rate 0.10 \
      --required-oracle-arm two_pulse \
      --macro-steps 18 \
      --pulse-steps 2 \
      --three-pulse-beam-width 8
  fi
  echo "CUDA preflight attempt ${attempt}/${ATTEMPTS} failed; no CPU fallback will be used." >&2
done

echo "CUDA preflight failed after ${ATTEMPTS} attempts; PushT feasibility preflight was not run." >&2
exit 1
