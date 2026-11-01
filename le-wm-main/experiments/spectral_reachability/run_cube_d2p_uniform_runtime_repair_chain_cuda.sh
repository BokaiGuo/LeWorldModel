#!/usr/bin/env bash
# Build an exact-P1-candidate runtime repair, then test it on unseen CUDA roots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
RECEIPT="${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json"
POOL="${ROOT}/results/cube_d2p_live_runtime_repair_pool_v7_uniform_1024_20261025_cuda.h5"
CHECKPOINT="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm_d2p_repair_v5_runtime_uniform_pool"
P1_ROOT="${ROOT}/results/cube_d2p_repair_v5_runtime_uniform_pool_fresh_p1_20261025_cuda"
BASE="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm_d2p_repair_v3"
PYTHON="${ROOT}/.venv/bin/python"

test -f "${SOURCE}"
test -f "${RECEIPT}"
test -f "${BASE}/weights.pt"
test ! -e "${POOL}"
test ! -e "${CHECKPOINT}"
test ! -e "${P1_ROOT}"

export MUJOCO_GL=egl
"${PYTHON}" -m experiments.spectral_reachability.build_cube_d2p_runtime_repair_pool \
  --output "${POOL}" --receipt "${RECEIPT}" --training-dataset "${SOURCE}" \
  --samples 1024 --seed 202613501 --candidate-distribution uniform \
  --candidate-seed 202613502 --device cuda

"${PYTHON}" -m experiments.spectral_reachability.train_cube_d2p_runtime_aligned_repair \
  --dataset "${SOURCE}" --receipt "${RECEIPT}" --runtime-pool "${POOL}" \
  --runtime-pool-receipt "${POOL}.receipt.json" --base-checkpoint "${BASE}" \
  --output "${CHECKPOINT}" --epochs 12 --batches 256 --validation-batches 128 \
  --batch-size 32 --learning-rate 2e-5 --seed 202613601 --alignment-samples 4 --device cuda

"${PYTHON}" - "${CHECKPOINT}/repair_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") != "trained_runtime_aligned_repair":
    raise SystemExit("uniform-candidate checkpoint is incomplete")
if manifest.get("training_contract", {}).get("candidate_distribution") != "uniform":
    raise SystemExit("uniform-candidate repair lost its P1 candidate-law binding")
PY

"${ROOT}/experiments/spectral_reachability/run_cube_d2p_runtime_repair_p1_chain_cuda.sh" \
  "${CHECKPOINT}" "${P1_ROOT}" 202618000
