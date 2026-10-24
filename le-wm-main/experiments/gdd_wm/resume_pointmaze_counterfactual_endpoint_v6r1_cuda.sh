#!/usr/bin/env bash
# Resume only the CUDA endpoint stage of an immutable V6R1 panel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
VALIDATOR="experiments.gdd_wm.validate_pointmaze_counterfactual_endpoint_v6_panel"
RUNNER="experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control"
AUDIT="experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"

PREFLIGHT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_static_preflight_20261023.json"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_fresh_20261023_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_fresh_endpoint_20261023_cuda"

for required in "${PYTHON}" "${WRAPPER}" "${PREFLIGHT}" "${PANEL}"; do
  [[ -e "${required}" ]] || { echo "missing V6R1 resume input: ${required}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite V6R1 endpoint output" >&2; exit 4; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 5; }

"${PYTHON}" -m "${VALIDATOR}" --panel "${PANEL}" --frozen-preflight "${PREFLIGHT}"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202612005 --width 32 --steps 400 --ensemble-size 4 --risk-alpha 0.1 \
  --minimum-summary-collision-pairs-per-root 8

"${PYTHON}" -m "${AUDIT}" --report "${OUTPUT}/report.json" --output "${OUTPUT}/contract_audit.json"
bash "${WRAPPER}" "${PYTHON}" -m "${ALIAS_AUDIT}" --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" --outcome-report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05
"${PYTHON}" -m "${DISAGREEMENT_AUDIT}" --report "${OUTPUT}/report.json" --left-arm cost --right-arm endpoint \
  --output "${OUTPUT}/cost_endpoint_disagreement_audit.json"
