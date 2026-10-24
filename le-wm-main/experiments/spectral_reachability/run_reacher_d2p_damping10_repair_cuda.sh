#!/usr/bin/env bash
# CUDA-only fixed-point continuation after a damping=1 repair selects damping=10.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
BASE="${ROOT}/results/reacher_d2p_damping_matched_repair_v1_20261024_cuda"
RETENTION="${ROOT}/results/reacher_d2p_joint_repair_v1_20261024_cuda"
RUN="${ROOT}/results/reacher_d2p_damping10_repair_v1_20261024_cuda"
DATASET="$(jq -er '.dataset' "${ROOT}/data/reacher_official_source_20261024/official_source_manifest.json")"
[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite ${RUN}" >&2; exit 2; }
[[ -f "${BASE}/checkpoint/repair_manifest.json" && -f "${BASE}/p0_damping_selection.json" && -f "${RETENTION}/counterfactual_pool/manifest.json" ]] || exit 2
D="$(jq -er '.selected.damping' "${BASE}/p0_damping_selection.json")"
[[ "${D}" == "10" || "${D}" == "10.0" ]] || { echo "expected the sealed base damping=10 continuation" >&2; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "${RUN}"; cp "${BASE}/stats.json" "${RUN}/stats.json"
.venv/bin/python -m experiments.spectral_reachability.build_reacher_d2p_p1_proposal_pair_pool --output "${RUN}/counterfactual_pool" --policy "${BASE}/checkpoint" --adapter-artifact "${RUN}/stats.json" --seed 20272500 --roots 96 --damping "${D}" --proposals 128 --device cuda
.venv/bin/python -m experiments.spectral_reachability.train_reacher_d2p_joint_repair --dataset "${DATASET}" --pair-pool "${RUN}/counterfactual_pool/pairs.h5" --pair-manifest "${RUN}/counterfactual_pool/manifest.json" --retention-pair-pool "${RETENTION}/counterfactual_pool/pairs.h5" --retention-pair-manifest "${RETENTION}/counterfactual_pool/manifest.json" --adapter-artifact "${RUN}/stats.json" --base-checkpoint "${BASE}/checkpoint" --output "${RUN}/checkpoint" --seed 20272601 --epochs 8 --transition-batch-size 32 --pair-batch-size 8 --steps-per-epoch 128 --learning-rate 1e-5 --counterfactual-weight 1 --device cuda
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20272800 --roots 4 --output "${RUN}/p0_preflight"
for d in 0.001 0.01 0.1 1 10; do .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20272900 --roots 4 --calibration-roots 4 --calibration-only --damping "$d" --output "${RUN}/p0_calibration_d${d}"; done
.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping --reports "${RUN}/p0_calibration_d0.001/report.json" "${RUN}/p0_calibration_d0.01/report.json" "${RUN}/p0_calibration_d0.1/report.json" "${RUN}/p0_calibration_d1/report.json" "${RUN}/p0_calibration_d10/report.json" --output "${RUN}/p0_damping_selection.json"
D_NEXT="$(jq -er '.selected.damping' "${RUN}/p0_damping_selection.json")"
[[ "${D_NEXT}" == "${D}" ]] || { echo "P1 source/deployment damping changed; rebuild pool before P1" >&2; exit 5; }
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20273000 --roots 24 --calibration-roots 0 --outcome-only --damping "$D_NEXT" --calibration-evidence "${RUN}/p0_damping_selection.json" --output "${RUN}/p0_outcome"
[[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN}/p0_outcome/report.json")" == true ]] || { echo "P0 sealed" >&2; exit 3; }
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20273100 --roots 4 --p0-evidence "${RUN}/p0_outcome/report.json" --damping "$D_NEXT" --output "${RUN}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20273200 --roots 6 --calibration-only --p0-evidence "${RUN}/p0_outcome/report.json" --p1-preflight-evidence "${RUN}/p1_preflight" --damping "$D_NEXT" --mechanism-observability observed_encoded_goal_cost --training-source-manifest "${RUN}/counterfactual_pool/manifest.json" --training-source-manifest "${RETENTION}/counterfactual_pool/manifest.json" --output "${RUN}/p1_calibration"
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --report "${RUN}/p1_calibration/report.json" --output "${RUN}/p1_fidelity.json"
jq -e '(.summary.model_to_observation_goal_order_agreement_count == .summary.eligible_roots_audited) and ([.gates[]]|all)' "${RUN}/p1_fidelity.json" >/dev/null || { echo "P1 sealed" >&2; exit 4; }
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --device cuda --seed 20273300 --roots 12 --outcome-only --calibration-evidence "${RUN}/p1_calibration/report.json" --p0-evidence "${RUN}/p0_outcome/report.json" --p1-preflight-evidence "${RUN}/p1_preflight" --damping "$D_NEXT" --mechanism-observability observed_encoded_goal_cost --training-source-manifest "${RUN}/counterfactual_pool/manifest.json" --training-source-manifest "${RETENTION}/counterfactual_pool/manifest.json" --output "${RUN}/p1_outcome"
