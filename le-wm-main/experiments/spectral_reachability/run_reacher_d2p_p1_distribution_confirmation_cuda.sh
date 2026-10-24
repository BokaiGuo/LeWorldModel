#!/usr/bin/env bash
# Independent fixed-contract confirmation after an inconclusive P0-vs-CEM
# interval.  No previous outcome root is reused and no hyperparameter changes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
SOURCE_RUN="${1:-${ROOT}/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda}"
RUN_ROOT="${2:-${ROOT}/results/reacher_d2p_p1_distribution_confirmation_v1_20261024_cuda}"
SOURCE_RUN="$(cd "$(dirname "${SOURCE_RUN}")" && pwd)/$(basename "${SOURCE_RUN}")"
RUN_ROOT="$(cd "$(dirname "${RUN_ROOT}")" && pwd)/$(basename "${RUN_ROOT}")"
CHECKPOINT="${SOURCE_RUN}/checkpoint"
STATS="${SOURCE_RUN}/reacher_training_action_stats.json"
SELECTION="${SOURCE_RUN}/p0_damping_selection.json"
SOURCE_P0="${SOURCE_RUN}/p0_outcome/report.json"

[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to overwrite: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CHECKPOINT}/repair_manifest.json" && -f "${STATS}" && -f "${SELECTION}" && -f "${SOURCE_P0}" ]] || { echo "required sealed source evidence is missing" >&2; exit 2; }
jq -e '(.stage_gate.p1_unlocked == false) and (.protocol.root_disjoint_calibration_outcome == true)' "${SOURCE_P0}" >/dev/null
DAMPING="$(jq -er '.selected.damping' "${SELECTION}")"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "${RUN_ROOT}"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20270300 --roots 24 --calibration-roots 0 --outcome-only --damping "${DAMPING}" --calibration-evidence "${SELECTION}" --output "${RUN_ROOT}/p0_confirmation_n24"
if [[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN_ROOT}/p0_confirmation_n24/report.json")" != "true" ]]; then
  echo "independent P0 confirmation did not unlock P1; preserve the sealed result and stop." >&2
  exit 3
fi
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20270400 --roots 4 --p0-evidence "${RUN_ROOT}/p0_confirmation_n24/report.json" --damping "${DAMPING}" --output "${RUN_ROOT}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20270500 --roots 6 --calibration-only --p0-evidence "${RUN_ROOT}/p0_confirmation_n24/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --training-source-manifest "${CHECKPOINT}/repair_manifest.json" --output "${RUN_ROOT}/p1_calibration"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --report "${RUN_ROOT}/p1_calibration/report.json" --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"
if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]] || ! jq -e '(.status == "completed") and (.summary.eligible_roots_audited > 0) and (.summary.model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited) and ([.gates[]] | all)' "${RUN_ROOT}/p1_first_chunk_fidelity.json" >/dev/null; then
  echo "P1 fidelity did not pass on the independent calibration roots; outcome remains sealed." >&2
  exit 4
fi
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20270600 --roots 12 --outcome-only --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" --p0-evidence "${RUN_ROOT}/p0_confirmation_n24/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --training-source-manifest "${CHECKPOINT}/repair_manifest.json" --output "${RUN_ROOT}/p1_outcome"
