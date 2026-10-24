#!/usr/bin/env bash
# Resume only the untouched P1 half after P0 succeeded but an incompatible
# preflight damping was caught before P1 calibration created any roots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
RUN_ROOT_INPUT="${1:?usage: resume_reacher_d2p_joint_p1_cuda.sh RUN_ROOT}"
RUN_ROOT="$(cd "$(dirname "${RUN_ROOT_INPUT}")" && pwd)/$(basename "${RUN_ROOT_INPUT}")"
CHECKPOINT="${RUN_ROOT}/checkpoint"
STATS="${RUN_ROOT}/reacher_training_action_stats.json"
P0="${RUN_ROOT}/p0_outcome/report.json"
PREFLIGHT="${RUN_ROOT}/p1_preflight_selected_damping"

if [[ ! -f "${CHECKPOINT}/repair_manifest.json" || ! -f "${STATS}" || ! -f "${P0}" ]]; then
  echo "joint checkpoint, stats receipt, or P0 outcome is missing" >&2
  exit 2
fi
if [[ -e "${PREFLIGHT}" || -e "${RUN_ROOT}/p1_calibration" || -e "${RUN_ROOT}/p1_outcome" ]]; then
  echo "refusing to overwrite any selected-damping P1 evidence" >&2
  exit 2
fi
jq -e '(.stage_gate.p1_unlocked == true) and (.gates.cuda_only == true)' "${P0}" >/dev/null
DAMPING="$(jq -er '.selected.damping' "${RUN_ROOT}/p0_damping_selection.json")"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269410 --roots 4 --p0-evidence "${P0}" --damping "${DAMPING}" --output "${PREFLIGHT}"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269500 --roots 6 --calibration-only --p0-evidence "${P0}" --p1-preflight-evidence "${PREFLIGHT}" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_calibration"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --report "${RUN_ROOT}/p1_calibration/report.json" --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"
if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]] || ! jq -e '(.status == "completed") and (.summary.eligible_roots_audited > 0) and (.summary.model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited) and ([.gates[]] | all)' "${RUN_ROOT}/p1_first_chunk_fidelity.json" >/dev/null; then
  echo "joint repair did not preserve every first-chunk model-to-observation order; outcome roots remain sealed." >&2
  exit 4
fi
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20269600 --roots 12 --outcome-only --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" --p0-evidence "${P0}" --p1-preflight-evidence "${PREFLIGHT}" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_outcome"
