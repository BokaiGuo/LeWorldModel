#!/usr/bin/env bash
# Host-CUDA, read-only outcome audit; never runs inside Codex sandbox.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"
for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
  [[ -c "${node}" ]] || { echo "host CUDA device node is unavailable: ${node}" >&2; exit 3; }
done
[[ -x "${PYTHON}" ]] || { echo "missing project Python: ${PYTHON}" >&2; exit 5; }
"${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${PYTHON}" -m experiments.gdd_wm.audit_pointmaze_pairwise_contrast_v12_outcome
