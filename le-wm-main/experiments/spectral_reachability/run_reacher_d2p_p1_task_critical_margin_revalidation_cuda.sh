#!/usr/bin/env bash
# Host-CUDA-only calibration chain for the Reacher P1 task-coordinate repair.
#
# The planner remains frozen and RGB-only.  qpos distance is evaluator-only:
# it selects a calibration interval and verifies physical non-greediness, but
# never enters action scoring, proposal ranking, or the frozen model.  Outcome
# roots remain unopened unless two disjoint calibration stages pass.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BASE_RUN="${1:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7}"
RUN_ROOT="${2:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7_task_critical_margin_revalidation_v1}"
SEED_BASE="${3:-20270600}"
POLICY="${BASE_RUN}/checkpoint"
ADAPTER="${BASE_RUN}/reacher_training_action_stats.json"
P0_EVIDENCE="${BASE_RUN}/p0_outcome/report.json"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to overwrite immutable P1 task-critical root: ${RUN_ROOT}" >&2
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
SOURCE_SEED="$((SEED_BASE + 100))"
BOUND_SEED="$((SEED_BASE + 200))"
OUTCOME_SEED="$((SEED_BASE + 300))"
PREFLIGHT="${RUN_ROOT}/p1_preflight"
SOURCE="${RUN_ROOT}/p1_task_critical_source_calibration"
SELECTION="${RUN_ROOT}/p1_task_critical_margin_band_selection.json"
BOUND="${RUN_ROOT}/p1_task_critical_bound_calibration"
OUTCOME="${RUN_ROOT}/p1_task_critical_outcome"

# This executes on the host, not the Codex sandbox.  Every Python module also
# requires --device cuda, so no CPU fallback exists.
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader
mkdir -p "${RUN_ROOT}"

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${PREFLIGHT_SEED}" --roots 4 --min-model-witness-roots 2 --p0-evidence "${P0_EVIDENCE}" --damping 10 \
  --output "${PREFLIGHT}"

# Stage A: source calibration is independent and cannot itself open outcomes.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${SOURCE_SEED}" --roots 16 --calibration-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --damping 10 --mechanism-observability qpos_distance --output "${SOURCE}"

.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_p1_task_critical_margin_band \
  --report "${SOURCE}/report.json" --minimum-passing-roots 4 --output "${SELECTION}"

MIN_MARGIN="$(jq -er '.selected.min_non_greedy_margin' "${SELECTION}")"
MAX_MARGIN="$(jq -er '.selected.max_non_greedy_margin' "${SELECTION}")"

# Stage B: bind the candidate interval on new qpos-evaluated roots.  A failure
# exits here; no outcome directory nor outcome root is opened.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${BOUND_SEED}" --roots 16 --calibration-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --damping 10 --mechanism-observability qpos_distance \
  --non-greedy-margin "${MIN_MARGIN}" --max-non-greedy-margin "${MAX_MARGIN}" \
  --margin-band-selection-evidence "${SELECTION}" --output "${BOUND}"

if [[ "$(jq -er '.gates.every_model_eligible_reconfiguration_is_physically_non_greedy_relative_to_zero_prefix' "${BOUND}/report.json")" != "true" ]]; then
  echo "task-critical calibration failed physical non-greediness; outcome roots remain sealed." >&2
  exit 4
fi

# Stage C is reachable only after the source and bound calibrations pass.
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${POLICY}" --adapter-artifact "${ADAPTER}" --device cuda \
  --seed "${OUTCOME_SEED}" --roots 24 --outcome-only --p0-evidence "${P0_EVIDENCE}" \
  --p1-preflight-evidence "${PREFLIGHT}" --calibration-evidence "${BOUND}/report.json" \
  --damping 10 --mechanism-observability qpos_distance \
  --non-greedy-margin "${MIN_MARGIN}" --max-non-greedy-margin "${MAX_MARGIN}" --output "${OUTCOME}"
