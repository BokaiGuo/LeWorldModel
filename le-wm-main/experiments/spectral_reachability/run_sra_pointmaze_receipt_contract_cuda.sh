#!/usr/bin/env bash
# Rebuild the exact MuJoCo-2.2 position-label contract used by receipt-bound SRA P1/P2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/results/sra_pointmaze_replay_mujoco220_receipt_contract_20261024_cuda}"
ROOTS="1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847,1848,1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,1860,1861,1862,1863"

# Use module execution so imports resolve from the project root.  The parent
# requires host CUDA; MuJoCo only supplies evaluator labels after that gate.
exec "${ROOT}/experiments/run_with_torch_cudnn.sh" "${ROOT}/.venv/bin/python" -m \
  experiments.spectral_reachability.audit_sra_pointmaze_replay_legacy_backend_cuda \
  --output-dir "${OUTPUT_DIR}" \
  --legacy-python "${ROOT}/.legacy_mujoco210_py310/bin/python" \
  --roots "${ROOTS}" \
  --engine mujoco \
  --allow-position-label-contract
