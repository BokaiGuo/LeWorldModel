#!/usr/bin/env bash
# CUDA-only entrypoint.  Invoke directly on the host (for example through
# systemd-run --user); the Python module rejects CPU fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
OUTPUT="${1:-${ROOT}/results/d2p_reacher_p1_reconfiguration_preflight_cuda}"
P0_EVIDENCE="${2:-${ROOT}/results/d2p_reacher_p0_snapshot_r4_confirmation3_n48_20261024_cuda/report.json}"

exec .venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --device cuda \
  --output "${OUTPUT}" \
  --p0-evidence "${P0_EVIDENCE}"
