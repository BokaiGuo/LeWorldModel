#!/usr/bin/env bash
# Idempotently close a V6R2 run after an interruption in post-execution audits.
# It never rewrites a panel, fits a CPU substitute, or reruns sealed outcomes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_panel_20261023_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_endpoint_20261023_cuda"
REPORT="${OUTPUT}/report.json"

for required in "${PYTHON}" "${WRAPPER}" "${PANEL}/temporal_ood_v6_context.json" "${REPORT}" "${OUTPUT}/pre_outcome_predictions.json"; do
  [[ -e "${required}" ]] || { echo "missing V6R2 resume input: ${required}" >&2; exit 3; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda

if [[ ! -e "${OUTPUT}/contract_audit.json" ]]; then
  "${PYTHON}" -m experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6 --report "${REPORT}" --output "${OUTPUT}/contract_audit.json"
fi
if [[ ! -e "${OUTPUT}/decision_aliasing_calibration_outcome.json" ]]; then
  bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit \
    --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
    --outcome-report "${REPORT}" --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" \
    --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05
fi
if [[ ! -e "${OUTPUT}/cost_endpoint_disagreement_audit.json" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.audit_counterfactual_arm_disagreement \
    --report "${REPORT}" --left-arm cost --right-arm endpoint --output "${OUTPUT}/cost_endpoint_disagreement_audit.json"
fi
if [[ ! -e "${OUTPUT}/decision_spectral_mismatch_audit.json" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.audit_pointmaze_decision_spectral_mismatch \
    --report "${REPORT}" --output "${OUTPUT}/decision_spectral_mismatch_audit.json"
fi

expected_hash="$(jq -r '.pre_outcome_prediction_bundle.sha256' "${REPORT}")"
actual_hash="$(sha256sum "${OUTPUT}/pre_outcome_predictions.json" | awk '{print $1}')"
[[ "${expected_hash}" == "${actual_hash}" ]] || { echo "sealed prediction-bundle hash mismatch" >&2; exit 5; }
jq -e '
  .status == "completed_counterfactual_endpoint_control" and
  .split_isolation.root_disjoint == true and
  .split_isolation.outcome_labels_sealed_before_execution == true and
  .control_contract.terminal_endpoint_predictions_sealed_before_outcome_execution == true and
  .cuda_preflight.device == "cuda"
' "${REPORT}" >/dev/null
jq -e '.status == "passed_contract_audit"' "${OUTPUT}/contract_audit.json" >/dev/null
jq -e '.status == "completed_fixed_metric_audit" and .protocol.metric_selected_before_outcomes == true' \
  "${OUTPUT}/decision_spectral_mismatch_audit.json" >/dev/null
echo "V6R2_RESUME_COMPLETE"
