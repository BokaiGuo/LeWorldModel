#!/usr/bin/env bash
# Fresh CUDA-only P1 calibration for a explicitly labelled runtime repair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?usage: $0 /absolute/runtime-repair-checkpoint /absolute/run-root}"
RUN_ROOT="${2:?usage: $0 /absolute/runtime-repair-checkpoint /absolute/run-root}"
P1_SEED="${3:-202612000}"
P1_ROOTS="${4:-8}"
[[ "${P1_ROOTS}" =~ ^[1-9][0-9]*$ ]] || { echo "P1 roots must be a positive integer" >&2; exit 2; }
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
RECEIPT="${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json"
PYTHON="${ROOT}/.venv/bin/python"

test -f "${CHECKPOINT}/weights.pt"
test -f "${CHECKPOINT}/repair_manifest.json"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

"${PYTHON}" - "${CHECKPOINT}/repair_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") not in {"trained_runtime_aligned_repair", "trained_same_root_ranking_repair"}:
    raise SystemExit("checkpoint is not a complete runtime-aligned or same-root-ranking repair")
if manifest.get("training_contract", {}).get("mode") not in {"runtime_pool", "same_root_ranking_pool"}:
    raise SystemExit("checkpoint is not trained on an admissible runtime repair pool")
if manifest.get("training_contract", {}).get("cpu_fallback") is not False:
    raise SystemExit("runtime repair manifest is not CUDA-only")
PY

"${PYTHON}" -m experiments.spectral_reachability.cube_d2p_repair_root_admission \
  --checkpoint "${CHECKPOINT}" --seed "${P1_SEED}" --roots "${P1_ROOTS}" \
  --output "${RUN_ROOT}/root_admission.json"

export MUJOCO_GL=egl
"${PYTHON}" -m experiments.spectral_reachability.build_cube_d2p_live_runtime_panel \
  --output "${RUN_ROOT}/p1_live_roots" \
  --training-dataset "${SOURCE}" \
  --training-source-receipt "${RECEIPT}" \
  --roots "${P1_ROOTS}" --seed "${P1_SEED}" --device cuda

"${PYTHON}" -m experiments.spectral_reachability.preflight_cube_d2p_p1_live_panel \
  --output "${RUN_ROOT}/p1_model_side" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --training-source-receipt "${RECEIPT}" \
  --policy "${CHECKPOINT}" --proposals 16 --seed "$((P1_SEED + 1))" --device cuda

"${PYTHON}" -m experiments.spectral_reachability.run_cube_d2p_p1_live_physical_calibration \
  --output "${RUN_ROOT}/p1_physical_calibration" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --p1-evidence "${RUN_ROOT}/p1_model_side/report.json" --device cuda

"${PYTHON}" -m experiments.spectral_reachability.audit_cube_d2p_p1_visual_rank \
  --output "${RUN_ROOT}/p1_visual_rank_audit" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --p1-evidence "${RUN_ROOT}/p1_model_side/report.json" \
  --policy "${CHECKPOINT}" --device cuda
