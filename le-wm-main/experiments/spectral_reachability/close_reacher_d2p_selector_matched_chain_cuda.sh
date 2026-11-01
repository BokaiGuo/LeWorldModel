#!/usr/bin/env bash
# Continue a selector-stable low-drift repair through fresh CUDA P0/P1 gates.
# Every evidence path is new; any unmet gate stops the chain before outcomes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
RUN="${1:?usage: $0 SELECTOR_STABLE_REPAIR_RUN}"
RUN="$(cd "${RUN}" && pwd)/$(basename "${RUN}")"
CHECKPOINT="${RUN}/checkpoint"
STATS="${RUN}/stats.json"
POOL_MANIFEST="${RUN}/../selector_matched_d1_pool_fullstate_20261025_cuda/manifest.json"
SELECTOR="${RUN}/selector_stability/report.json"

[[ -f "${CHECKPOINT}/weights.pt" && -f "${STATS}" && -f "${POOL_MANIFEST}" && -f "${SELECTOR}" ]] || {
  echo "selector-stable repair inputs are incomplete" >&2; exit 2;
}
jq -e '(.status == "completed") and (.selection.all_source_selections_stable == true)' "${SELECTOR}" >/dev/null
jq -e '(.schema == "reacher_d2p_p1_proposal_pair_pool.v1") and (.proposal_contract.damping == 1)' "${POOL_MANIFEST}" >/dev/null
for path in p0_preflight_fullclosure p0_damping_selection_fullclosure.json p0_outcome_fullclosure p1_preflight_fullclosure p1_calibration_fullclosure p1_fidelity_fullclosure.json p1_outcome_fullclosure; do
  [[ ! -e "${RUN}/${path}" ]] || { echo "refusing to overwrite ${RUN}/${path}" >&2; exit 2; }
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272200 --roots 4 \
  --output "${RUN}/p0_preflight_fullclosure"
for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272300 --roots 4 \
    --calibration-roots 4 --calibration-only --damping "${damping}" --output "${RUN}/p0_calibration_fullclosure_d${damping}"
done
.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports "${RUN}/p0_calibration_fullclosure_d0.001/report.json" "${RUN}/p0_calibration_fullclosure_d0.01/report.json" \
  "${RUN}/p0_calibration_fullclosure_d0.1/report.json" "${RUN}/p0_calibration_fullclosure_d1/report.json" \
  "${RUN}/p0_calibration_fullclosure_d10/report.json" --output "${RUN}/p0_damping_selection_fullclosure.json"
DAMPING="$(jq -er '.selected.damping' "${RUN}/p0_damping_selection_fullclosure.json")"
[[ "${DAMPING}" == "1" || "${DAMPING}" == "1.0" ]] || {
  echo "fresh P0 selected damping ${DAMPING}, outside the selector-matched P1 pool contract; rebuild the pool" >&2; exit 4;
}
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272400 --roots 24 \
  --calibration-roots 0 --outcome-only --damping "${DAMPING}" \
  --calibration-evidence "${RUN}/p0_damping_selection_fullclosure.json" --output "${RUN}/p0_outcome_fullclosure"
jq -e '.stage_gate.p1_unlocked == true' "${RUN}/p0_outcome_fullclosure/report.json" >/dev/null || {
  echo "fresh P0 did not unlock P1" >&2; exit 4;
}
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272500 --roots 4 \
  --p0-evidence "${RUN}/p0_outcome_fullclosure/report.json" --damping "${DAMPING}" --output "${RUN}/p1_preflight_fullclosure"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272600 --roots 6 --calibration-only \
  --p0-evidence "${RUN}/p0_outcome_fullclosure/report.json" --p1-preflight-evidence "${RUN}/p1_preflight_fullclosure" \
  --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost \
  --training-source-manifest "${POOL_MANIFEST}" --selector-stability-evidence "${SELECTOR}" \
  --output "${RUN}/p1_calibration_fullclosure"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --report "${RUN}/p1_calibration_fullclosure/report.json" \
  --output "${RUN}/p1_fidelity_fullclosure.json"
jq -e '([.gates[]] | all)' "${RUN}/p1_calibration_fullclosure/report.json" >/dev/null
jq -e '(.status == "completed") and ([.gates[]] | all)' "${RUN}/p1_fidelity_fullclosure.json" >/dev/null || {
  echo "fresh P1 calibration/fidelity failed; outcome roots remain sealed" >&2; exit 4;
}
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 20272700 --roots 12 --outcome-only \
  --calibration-evidence "${RUN}/p1_calibration_fullclosure/report.json" \
  --p0-evidence "${RUN}/p0_outcome_fullclosure/report.json" --p1-preflight-evidence "${RUN}/p1_preflight_fullclosure" \
  --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost \
  --training-source-manifest "${POOL_MANIFEST}" --selector-stability-evidence "${SELECTOR}" \
  --output "${RUN}/p1_outcome_fullclosure"
