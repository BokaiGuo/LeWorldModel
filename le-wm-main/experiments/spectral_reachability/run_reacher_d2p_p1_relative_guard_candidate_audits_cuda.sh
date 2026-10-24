#!/usr/bin/env bash
# CUDA-only candidate diagnostics for the fresh relative-V_reach-guard
# calibration roots.  These diagnostics are calibration analysis only: they
# expose physical consequences after a model-only top-k ordering is frozen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

P0_EVIDENCE="${ROOT}/results/d2p_reacher_p0_snapshot_r4_confirmation3_n48_20261024_cuda/report.json"
P1_PREFLIGHT="${ROOT}/results/d2p_reacher_p1_relative_guard_preflight_20261024_cuda"

for seed in 20263210 20263211 20263212 20263213 20263214 20263215; do
  output="${ROOT}/results/d2p_reacher_p1_relative_guard_candidate_prefix_audit_seed${seed}_20261024_cuda.json"
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
