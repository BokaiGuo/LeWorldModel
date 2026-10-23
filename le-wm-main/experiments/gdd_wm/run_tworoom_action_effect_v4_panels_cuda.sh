#!/usr/bin/env bash
# Three root-disjoint, host-CUDA replications of the action-effect comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_tworoom_action_effect_repairability"
AUDITOR="experiments.gdd_wm.audit_tworoom_action_effect_v4_panels"
OUTPUT="${ROOT}/results/tworoom_action_effect_repairability_v4_panels_20261023_cuda"
POLICY="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"
ADAPTER="${ROOT}/results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"

[[ ! -e "${OUTPUT}" ]] || { echo "immutable output exists: ${OUTPUT}" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${POLICY}" "${ADAPTER}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 3; }
done
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

mkdir -p "${OUTPUT}"
SEEDS=(20264416 20264576 20264736)
REPORTS=()
for index in "${!SEEDS[@]}"; do
  panel="${OUTPUT}/panel_$((index + 1))"
  bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
    --output "${panel}" --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
    --seed "${SEEDS[index]}" --roots 160 --pulse-steps 5 --decision-blocks 6 \
    --width 96 --epochs 180 --learning-rate 3e-3 --bootstrap-samples 10000
  REPORTS+=(--report "${panel}/report.json")
done
"${PYTHON}" -m "${AUDITOR}" "${REPORTS[@]}" --output "${OUTPUT}/v4_panel_audit.json"
