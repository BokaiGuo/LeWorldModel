#!/usr/bin/env bash
# End-to-end, host-CUDA-only counterfactual repair and sealed D2P re-evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
RUN_ROOT="${1:-${ROOT}/results/reacher_d2p_counterfactual_repair_20261024_cuda_v1}"
BASE="${2:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7/checkpoint}"
ADAPTER="${3:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7/reacher_training_action_stats.json}"

if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite immutable counterfactual repair run: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${BASE}/weights.pt" || ! -f "${ADAPTER}" ]]; then
  echo "base checkpoint or action-stat receipt is missing" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "${RUN_ROOT}"

.venv/bin/python -m experiments.spectral_reachability.build_reacher_d2p_counterfactual_pair_pool \
  --output "${RUN_ROOT}/counterfactual_pool" --seed 20267000 --episodes 128 --episode-steps 80 --device cuda
.venv/bin/python -m experiments.spectral_reachability.train_reacher_d2p_counterfactual_repair \
  --pool "${RUN_ROOT}/counterfactual_pool/pairs.h5" --pool-manifest "${RUN_ROOT}/counterfactual_pool/manifest.json" \
  --adapter-artifact "${ADAPTER}" --base-checkpoint "${BASE}" --output "${RUN_ROOT}/checkpoint" \
  --seed 20267001 --epochs 8 --batch-size 8 --learning-rate 1e-5 --ranking-temperature 0.05 --device cuda

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed 20267100 --roots 4 --output "${RUN_ROOT}/p0_preflight"
for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
    --seed 20267200 --roots 4 --calibration-roots 4 --calibration-only --damping "${damping}" \
    --output "${RUN_ROOT}/p0_calibration_d${damping}"
done
.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports "${RUN_ROOT}"/p0_calibration_d*/report.json --output "${RUN_ROOT}/p0_damping_selection.json"
DAMPING="$(jq -er '.selected.damping' "${RUN_ROOT}/p0_damping_selection.json")"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed 20267300 --roots 12 --calibration-roots 0 --outcome-only --damping "${DAMPING}" \
  --calibration-evidence "${RUN_ROOT}/p0_damping_selection.json" --output "${RUN_ROOT}/p0_outcome"
if [[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN_ROOT}/p0_outcome/report.json")" != "true" ]]; then
  echo "P0 did not unlock P1; preserve this sealed outcome and stop." >&2
  exit 3
fi

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed 20267400 --roots 4 --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --output "${RUN_ROOT}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed 20267500 --roots 6 --calibration-only --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_calibration"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" \
  --report "${RUN_ROOT}/p1_calibration/report.json" --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"
if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]] || \
   ! jq -e '(.status == "completed") and (.summary.eligible_roots_audited > 0) and (.summary.model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited) and ([.gates[]] | all)' "${RUN_ROOT}/p1_first_chunk_fidelity.json" >/dev/null; then
  echo "counterfactual calibration did not preserve all model-to-observation first-chunk orders; outcome roots remain sealed." >&2
  exit 4
fi
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed 20267600 --roots 12 --outcome-only --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" \
  --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" \
  --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_outcome"
