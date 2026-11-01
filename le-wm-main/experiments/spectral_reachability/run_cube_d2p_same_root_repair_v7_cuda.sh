#!/usr/bin/env bash
# Training-pool-only early stopping; never tune from a P1 outcome or relax gates.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
CHECKPOINT="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm_d2p_repair_v7_same_root_singleton"
RUN_ROOT="${ROOT}/results/cube_d2p_repair_v7_same_root_singleton_p1_20261025_cuda"
POOL="${ROOT}/results/cube_d2p_same_root_ranking_pool_v1_256x9_20261025_cuda.h5"
test ! -e "${CHECKPOINT}"
test ! -e "${RUN_ROOT}"
export MUJOCO_GL=egl
"${PYTHON}" -m experiments.spectral_reachability.train_cube_d2p_same_root_ranking_repair \
  --dataset "${ROOT}/../data/cube_source_20261025/cube_single_expert.h5" \
  --receipt "${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json" \
  --pool "${POOL}" --pool-receipt "${POOL}.receipt.json" \
  --base-checkpoint /home/gbk/.stable_worldmodel/checkpoints/cube/lewm_d2p_repair_v3 \
  --output "${CHECKPOINT}" --epochs 96 --patience 12 --batch-size 16 \
  --learning-rate 2e-5 --seed 202616100 --device cuda
"${PYTHON}" -m experiments.spectral_reachability.audit_cube_d2p_same_root_runtime_alignment \
  --checkpoint "${CHECKPOINT}" \
  --output "${ROOT}/results/cube_d2p_repair_v7_same_root_singleton_alignment_20261025_cuda.json" --device cuda
/bin/bash "${ROOT}/experiments/spectral_reachability/run_cube_d2p_runtime_repair_p1_chain_cuda.sh" \
  "${CHECKPOINT}" "${RUN_ROOT}" 202619000
