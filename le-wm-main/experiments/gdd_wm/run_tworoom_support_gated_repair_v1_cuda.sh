#!/usr/bin/env bash
# Host-CUDA C2 test: calibration-frozen support gate versus both components.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_tworoom_action_effect_repairability"
AUDITOR="experiments.gdd_wm.audit_tworoom_support_gated_repair_v1"
OUTPUT="${ROOT}/results/tworoom_support_gated_repair_v1_20261023_cuda"
POLICY="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"
ADAPTER="${ROOT}/results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"

[[ ! -e "${OUTPUT}" ]] || { echo "immutable output exists: ${OUTPUT}" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${POLICY}" "${ADAPTER}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 3; }
done
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

mkdir -p "${OUTPUT}"
SEEDS=(20264896 20265056 20265216)
REPORTS=()
for index in "${!SEEDS[@]}"; do
  panel="${OUTPUT}/panel_$((index + 1))"
  bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
    --output "${panel}" --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
    --seed "${SEEDS[index]}" --roots 160 --pulse-steps 5 --decision-blocks 6 \
    --width 96 --epochs 180 --learning-rate 3e-3 --bootstrap-samples 10000 \
    --support-gated-repair --support-gate-minimum-coverage 0.10
  REPORTS+=(--report "${panel}/report.json")
done
"${PYTHON}" -m "${AUDITOR}" "${REPORTS[@]}" --output "${OUTPUT}/support_gated_outcome_audit.json"
