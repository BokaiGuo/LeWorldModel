#!/usr/bin/env bash
# Run only after train_cube_d2p_dynamics_repair.py has atomically produced a
# complete, source-only repaired checkpoint. Every stage is host CUDA only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?usage: $0 /absolute/repaired-checkpoint /absolute/run-root}"
RUN_ROOT="${2:?usage: $0 /absolute/repaired-checkpoint /absolute/run-root}"
SOURCE="${ROOT}/../data/cube_source_20261025/cube_single_expert.h5"
RECEIPT="${ROOT}/results/cube_d2p_training_source_20261025/cube_training_source.json"
PYTHON="${ROOT}/.venv/bin/python"

test -f "${CHECKPOINT}/weights.pt"
test -f "${CHECKPOINT}/config.json"
test -f "${CHECKPOINT}/repair_manifest.json"
test ! -e "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

"${PYTHON}" - "${CHECKPOINT}/repair_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "trained_train_source_only":
    raise SystemExit("repair checkpoint does not have a complete source-only training manifest")
if report.get("training_contract", {}).get("cpu_fallback") is not False:
    raise SystemExit("repair manifest is not CUDA-only")
if report.get("episode_split", {}).get("overlap") is not False:
    raise SystemExit("repair manifest has overlapping train/validation episodes")
PY

export MUJOCO_GL=egl
"${PYTHON}" -m experiments.spectral_reachability.preflight_cube_d2p_p0 \
  --output "${RUN_ROOT}/p0_source_native" \
  --training-dataset "${SOURCE}" \
  --training-source-receipt "${RECEIPT}" \
  --policy "${CHECKPOINT}" \
  --device cuda \
  --roots 8 \
  --seed 202610311

"${PYTHON}" -m experiments.spectral_reachability.build_cube_d2p_live_runtime_panel \
  --output "${RUN_ROOT}/p1_live_roots" \
  --training-dataset "${SOURCE}" \
  --training-source-receipt "${RECEIPT}" \
  --roots 8 \
  --seed 202610312 \
  --device cuda

"${PYTHON}" -m experiments.spectral_reachability.preflight_cube_d2p_p1_live_panel \
  --output "${RUN_ROOT}/p1_model_side" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --training-source-receipt "${RECEIPT}" \
  --policy "${CHECKPOINT}" \
  --proposals 16 \
  --seed 202610313 \
  --device cuda

"${PYTHON}" -m experiments.spectral_reachability.run_cube_d2p_p1_live_physical_calibration \
  --output "${RUN_ROOT}/p1_physical_calibration" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --p1-evidence "${RUN_ROOT}/p1_model_side/report.json" \
  --device cuda

"${PYTHON}" -m experiments.spectral_reachability.audit_cube_d2p_p1_full_bank \
  --output "${RUN_ROOT}/p1_full_bank_rank_audit" \
  --live-panel "${RUN_ROOT}/p1_live_roots" \
  --p1-evidence "${RUN_ROOT}/p1_model_side/report.json" \
  --device cuda
