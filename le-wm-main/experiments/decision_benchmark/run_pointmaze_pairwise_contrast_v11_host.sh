#!/usr/bin/env bash
# Host-only CUDA entrypoint.  Codex Desktop's user namespace cannot expose
# /dev/nvidia*, so invoke this via `systemd-run --user` on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
cd "${ROOT}"

for node in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
  [[ -c "${node}" ]] || { echo "host CUDA device node is unavailable: ${node}" >&2; exit 3; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is unavailable; refusing CPU fallback" >&2; exit 4; }
[[ -x "${PYTHON}" ]] || { echo "missing project Python: ${PYTHON}" >&2; exit 5; }

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-cell-smoke
exec "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_pairwise_contrast_v11_cuda
