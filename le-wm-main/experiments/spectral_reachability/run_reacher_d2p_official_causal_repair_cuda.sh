#!/usr/bin/env bash
# Definitive host-CUDA-only causal-dynamics repair using the public Reacher
# training archive.  The source training set is disjoint from fresh P0/P1
# roots; every gate below is fail-closed and no CPU fallback exists.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SOURCE_ROOT="${1:-${ROOT}/data/reacher_official_source_20261024}"
RUN_ROOT="${2:-${ROOT}/results/reacher_d2p_official_causal_repair_20261024_cuda}"
SOURCE_MANIFEST="${SOURCE_ROOT}/official_source_manifest.json"
TRAINING_STATS="${RUN_ROOT}/reacher_training_action_stats.json"

if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite immutable official-source repair run: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "official dataset must first pass acquire_reacher_official_dataset.sh: ${SOURCE_MANIFEST}" >&2
  exit 2
fi

DATASET="$(jq -er '.dataset' "${SOURCE_MANIFEST}")"
if [[ ! -s "${DATASET}" ]]; then
  echo "validated official HDF5 is unavailable: ${DATASET}" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "${RUN_ROOT}"

.venv/bin/python -m experiments.spectral_reachability.build_reacher_d2p_training_action_stats \
  --dataset "${DATASET}" --source-manifest "${SOURCE_MANIFEST}" --output "${TRAINING_STATS}"

.venv/bin/python -m experiments.spectral_reachability.train_reacher_d2p_dynamics_repair \
  --dataset "${DATASET}" \
  --adapter-artifact "${TRAINING_STATS}" \
  --output "${RUN_ROOT}/checkpoint" \
  --device cuda \
  --seed 20266001 \
  --epochs 12 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --ranking-temperature 0.1 \
  --train-batches-per-epoch 512 --validation-batches 128

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
  --seed 20266100 --roots 4 --output "${RUN_ROOT}/p0_preflight"

for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
    --seed 20266200 --roots 4 --calibration-roots 4 --calibration-only --damping "${damping}" \
    --output "${RUN_ROOT}/p0_calibration_d${damping}"
done

.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports "${RUN_ROOT}/p0_calibration_d0.001/report.json" "${RUN_ROOT}/p0_calibration_d0.01/report.json" \
    "${RUN_ROOT}/p0_calibration_d0.1/report.json" "${RUN_ROOT}/p0_calibration_d1/report.json" \
    "${RUN_ROOT}/p0_calibration_d10/report.json" \
  --output "${RUN_ROOT}/p0_damping_selection.json"

DAMPING="$(jq -er '.selected.damping' "${RUN_ROOT}/p0_damping_selection.json")"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
  --seed 20266300 --roots 12 --calibration-roots 0 --outcome-only --damping "${DAMPING}" \
  --calibration-evidence "${RUN_ROOT}/p0_damping_selection.json" --output "${RUN_ROOT}/p0_outcome"

if [[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN_ROOT}/p0_outcome/report.json")" != "true" ]]; then
  echo "P0 did not unlock P1 for the official-source repaired checkpoint; stop before P1." >&2
  exit 3
fi

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
  --seed 20266400 --roots 4 --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --output "${RUN_ROOT}/p1_preflight"

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
  --seed 20266500 --roots 6 --calibration-only --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" --damping "${DAMPING}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN_ROOT}/p1_calibration"

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" \
  --report "${RUN_ROOT}/p1_calibration/report.json" --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"

if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]]; then
  echo "repaired checkpoint still fails the observed-successor calibration gate; outcome roots remain sealed." >&2
  exit 4
fi

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${RUN_ROOT}/checkpoint" --adapter-artifact "${TRAINING_STATS}" --device cuda \
  --seed 20266600 --roots 12 --outcome-only --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" \
  --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" \
  --damping "${DAMPING}" --mechanism-observability observed_encoded_goal_cost \
  --output "${RUN_ROOT}/p1_outcome"
