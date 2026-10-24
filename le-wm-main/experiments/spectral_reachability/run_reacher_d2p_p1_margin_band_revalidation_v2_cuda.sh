#!/usr/bin/env bash
# Host-CUDA-only revalidation after the P1 selector/preflight provenance fixes.
#
# The prior margin-band outcome remains diagnostic only: its selector did not
# bind the runtime geometry that generated the fresh calibration.  This script
# reserves new, disjoint roots for every stage and permits an outcome only if
# the current runner creates both the selector and the bound calibration.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BASE_RUN="${1:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7}"
RUN_ROOT="${2:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7_margin_band_revalidation_v4}"
SEED_BASE="${3:-20269800}"
POLICY="${BASE_RUN}/checkpoint"
ADAPTER="${BASE_RUN}/reacher_training_action_stats.json"
P0_EVIDENCE="${BASE_RUN}/p0_outcome/report.json"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite immutable P1 revalidation root: ${RUN_ROOT}" >&2
  exit 2
fi
for required in "${POLICY}/weights.pt" "${ADAPTER}" "${P0_EVIDENCE}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing required frozen evidence: ${required}" >&2
    exit 2
  fi
done
if ! [[ "${SEED_BASE}" =~ ^[0-9]+$ ]]; then
  echo "seed base must be a non-negative integer" >&2
  exit 2
fi
PREFLIGHT_SEED="${SEED_BASE}"
SOURCE_CALIBRATION_SEED="$((SEED_BASE + 100))"
BOUND_CALIBRATION_SEED="$((SEED_BASE + 200))"
OUTCOME_SEED="$((SEED_BASE + 300))"

# Deliberately executed only on the host service.  Every Python entry point
# independently rejects CPU, so this is an additional admission check rather
# than a fallback path.
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader

PREFLIGHT="${RUN_ROOT}/p1_preflight"
SOURCE_CALIBRATION="${RUN_ROOT}/p1_margin_source_calibration"
SOURCE_FIDELITY="${RUN_ROOT}/p1_margin_source_first_chunk_fidelity.json"
SELECTION="${RUN_ROOT}/p1_margin_band_selection.json"
BOUND_CALIBRATION="${RUN_ROOT}/p1_margin_band_calibration"
BOUND_FIDELITY="${RUN_ROOT}/p1_margin_band_first_chunk_fidelity.json"
OUTCOME="${RUN_ROOT}/p1_margin_band_outcome"

mkdir -p "${RUN_ROOT}"

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${PREFLIGHT_SEED}" --roots 4 --min-model-witness-roots 2 --p0-evidence "${P0_EVIDENCE}" --damping 10 \
  --output "${PREFLIGHT}"

# Stage A: new roots provide selector-only calibration.  No outcome roots are
# opened from this receipt.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${SOURCE_CALIBRATION_SEED}" --roots 16 --calibration-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --damping 10 \
  --mechanism-observability observed_encoded_goal_cost --output "${SOURCE_CALIBRATION}"

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" \
  --report "${SOURCE_CALIBRATION}/report.json" --output "${SOURCE_FIDELITY}"

.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_p1_margin_band_guard \
  --report "${SOURCE_CALIBRATION}/report.json" --fidelity "${SOURCE_FIDELITY}" \
  --minimum-passing-roots 4 --output "${SELECTION}"

MIN_MARGIN="$(jq -er '.selected.min_non_greedy_margin' "${SELECTION}")"
MAX_MARGIN="$(jq -er '.selected.max_non_greedy_margin' "${SELECTION}")"

# Stage B: the selector must pass a second, root-disjoint calibration under
# this exact runner, frozen checkpoint, target rendering, and planner geometry.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${BOUND_CALIBRATION_SEED}" --roots 16 --calibration-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --damping 10 \
  --non-greedy-margin "${MIN_MARGIN}" --max-non-greedy-margin "${MAX_MARGIN}" \
  --margin-band-selection-evidence "${SELECTION}" \
  --mechanism-observability observed_encoded_goal_cost --output "${BOUND_CALIBRATION}"

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" \
  --report "${BOUND_CALIBRATION}/report.json" --output "${BOUND_FIDELITY}"

if [[ "$(jq -er '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${BOUND_CALIBRATION}/report.json")" != "true" ]]; then
  echo "margin-band calibration did not preserve observed successor ordering; outcome roots remain sealed." >&2
  exit 4
fi

# Stage C: only an independently held-out panel can support a P1 efficacy
# claim.  It is bound to the Stage-B calibration, not to Stage-A selection.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${OUTCOME_SEED}" --roots 24 --outcome-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --calibration-evidence "${BOUND_CALIBRATION}/report.json" \
  --damping 10 --non-greedy-margin "${MIN_MARGIN}" --max-non-greedy-margin "${MAX_MARGIN}" \
  --mechanism-observability observed_encoded_goal_cost --output "${OUTCOME}"

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" \
  --report "${OUTCOME}/report.json" --output "${RUN_ROOT}/p1_margin_band_outcome_first_chunk_fidelity.json"
