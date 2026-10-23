#!/usr/bin/env bash
# Resume only the frozen V6 endpoint-control stage after a completed panel build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
PANEL_VALIDATOR="experiments.gdd_wm.validate_pointmaze_counterfactual_endpoint_v6_panel"
RUNNER="experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control"
CONTRACT_AUDIT="experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"
FROZEN_PREFLIGHT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6_static_preflight_20261021.json"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_endpoint_20261021_cuda"

if [[ ! -d "${PANEL}" ]]; then
  echo "missing completed immutable V6 panel: ${PANEL}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite V6 endpoint-control output: ${OUTPUT}" >&2
  exit 3
fi
for required in "${PYTHON}" "${WRAPPER}" "${FROZEN_PREFLIGHT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required V6 resume input: ${required}" >&2
    exit 4
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing CPU fallback" >&2
  exit 5
fi

# This read-only validator binds the already-built panel to the same ledger
# used before CUDA construction.  It never rebuilds roots, candidates, charts,
# or outcome labels.
"${PYTHON}" -m "${PANEL_VALIDATOR}" --panel "${PANEL}" --frozen-preflight "${FROZEN_PREFLIGHT}"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610340 --width 32 --steps 400 --ensemble-size 4 --risk-alpha 0.1 \
  --minimum-summary-collision-pairs-per-root 8

"${PYTHON}" -m "${CONTRACT_AUDIT}" --report "${OUTPUT}/report.json" --output "${OUTPUT}/contract_audit.json"
bash "${WRAPPER}" "${PYTHON}" -m "${ALIAS_AUDIT}" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --outcome-report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" \
  --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05
"${PYTHON}" -m "${DISAGREEMENT_AUDIT}" \
  --report "${OUTPUT}/report.json" --left-arm cost --right-arm endpoint \
  --output "${OUTPUT}/cost_endpoint_disagreement_audit.json"
