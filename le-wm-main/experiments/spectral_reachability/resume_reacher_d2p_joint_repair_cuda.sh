#!/usr/bin/env bash
# Continue only the sealed, source-only joint repair whose initial launcher
# stopped before creating any P0 root.  Every downstream output remains new.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
RUN_ROOT_INPUT="${1:?usage: resume_reacher_d2p_joint_repair_cuda.sh RUN_ROOT}"
RUN_ROOT="$(cd "$(dirname "${RUN_ROOT_INPUT}")" && pwd)/$(basename "${RUN_ROOT_INPUT}")"
STATS="${RUN_ROOT}/reacher_training_action_stats.json"
CHECKPOINT="${RUN_ROOT}/checkpoint"

if [[ ! -f "${CHECKPOINT}/weights.pt" || ! -f "${CHECKPOINT}/repair_manifest.json" || ! -f "${STATS}" ]]; then
  echo "joint source-only checkpoint, manifest, or training-stat receipt is missing" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}/p0_preflight" || -e "${RUN_ROOT}/p0_outcome" || -e "${RUN_ROOT}/p1_preflight" || -e "${RUN_ROOT}/p1_outcome" ]]; then
  echo "refusing to resume: this joint-repair run has already entered a root evaluation stage" >&2
  exit 2
fi
jq -e '(.schema == "reacher_d2p_joint_transition_counterfactual_repair.v1") and (.status == "trained_joint_source_only") and (.training_contract.p0_p1_roots_used == false) and (.training_contract.cpu_fallback == false)' "${CHECKPOINT}/repair_manifest.json" >/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269100 --roots 4 --output "${RUN_ROOT}/p0_preflight"
for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269200 --roots 4 --calibration-roots 4 --calibration-only --damping "${damping}" --output "${RUN_ROOT}/p0_calibration_d${damping}"
done
.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports "${RUN_ROOT}/p0_calibration_d0.001/report.json" "${RUN_ROOT}/p0_calibration_d0.01/report.json" "${RUN_ROOT}/p0_calibration_d0.1/report.json" "${RUN_ROOT}/p0_calibration_d1/report.json" "${RUN_ROOT}/p0_calibration_d10/report.json" --output "${RUN_ROOT}/p0_damping_selection.json"
DAMPING="$(jq -er '.selected.damping' "${RUN_ROOT}/p0_damping_selection.json")"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269300 --roots 12 --calibration-roots 0 --outcome-only --damping "${DAMPING}" --calibration-evidence "${RUN_ROOT}/p0_damping_selection.json" --output "${RUN_ROOT}/p0_outcome"
if [[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN_ROOT}/p0_outcome/report.json")" != "true" ]]; then
  echo "P0 did not unlock P1; preserve the sealed result and stop." >&2
  exit 3
fi
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269400 --roots 4 --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --damping "${DAMPING}" --output "${RUN_ROOT}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269500 --roots 6 --calibration-only --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_calibration"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --report "${RUN_ROOT}/p1_calibration/report.json" --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"
if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]] || ! jq -e '(.status == "completed") and (.summary.eligible_roots_audited > 0) and (.summary.model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited) and ([.gates[]] | all)' "${RUN_ROOT}/p1_first_chunk_fidelity.json" >/dev/null; then
  echo "joint repair did not preserve every first-chunk model-to-observation order; outcome roots remain sealed." >&2
  exit 4
fi
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269600 --roots 12 --outcome-only --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_outcome"
