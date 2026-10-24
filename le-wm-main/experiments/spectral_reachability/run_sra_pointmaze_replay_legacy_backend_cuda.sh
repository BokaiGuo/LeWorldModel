#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:?usage: $0 OUTPUT_DIR}"
LEGACY_PYTHON="${ROOT}/.legacy_mujoco210_py310/bin/python"
MUJOCO_RUNTIME="${ROOT}/.legacy_mujoco210_py310/vendor/mujoco-2.1.2"
exec "${ROOT}/experiments/run_with_torch_cudnn.sh" "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.audit_sra_pointmaze_replay_legacy_backend_cuda --output-dir "${OUTPUT_DIR}" --legacy-python "${LEGACY_PYTHON}" --engine mujoco_py --mujoco-py-path "${MUJOCO_RUNTIME}" --allow-position-label-contract
