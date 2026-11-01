#!/usr/bin/env bash
# Train an immutable CUDA-only runtime-aligned repair, then evaluate it only
# on fresh live P0/P1 roots.  No CPU fallback and no result-directory reuse.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?usage: $0 /absolute/new-checkpoint /absolute/new-run-root}"
RUN_ROOT="${2:?usage: $0 /absolute/new-checkpoint /absolute/new-run-root}"
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
RECEIPT="${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json"
PYTHON="${ROOT}/.venv/bin/python"

test ! -e "${CHECKPOINT}"
test ! -e "${RUN_ROOT}"
export MUJOCO_GL=egl

"${PYTHON}" -m experiments.spectral_reachability.train_cube_d2p_runtime_aligned_repair \
  --dataset "${SOURCE}" \
  --receipt "${RECEIPT}" \
  --output "${CHECKPOINT}" \
  --device cuda

exec "${ROOT}/experiments/spectral_reachability/run_cube_d2p_repair_p1_chain_cuda.sh" \
  "${CHECKPOINT}" "${RUN_ROOT}"
