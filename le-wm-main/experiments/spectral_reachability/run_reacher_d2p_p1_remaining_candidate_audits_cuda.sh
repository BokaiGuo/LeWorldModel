#!/usr/bin/env bash
# CUDA-only host entrypoint for the four outstanding P1 candidate-prefix
# diagnostics.  Keep the loop in this file: systemd expands dollar-prefixed
# text in an inline command, whereas it executes shell-script contents as-is.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

P0_EVIDENCE="${ROOT}/results/d2p_reacher_p0_snapshot_r4_confirmation3_n48_20261024_cuda/report.json"
P1_PREFLIGHT="${ROOT}/results/d2p_reacher_p1_reconfiguration_preflight_coverage128_20261024_cuda"

for seed in 20263101 20263103 20263104 20263105; do
  output="${ROOT}/results/d2p_reacher_p1_candidate_prefix_audit_seed${seed}_20261024_cuda.json"
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite immutable diagnostic output: ${output}" >&2
    exit 1
  fi
  .venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_candidate_prefixes \
    --output "${output}" \
    --p0-evidence "${P0_EVIDENCE}" \
    --p1-preflight-evidence "${P1_PREFLIGHT}" \
    --seed "${seed}" \
    --candidates 8
done
