#!/usr/bin/env bash
# Host-CUDA-only, fail-closed repair and fresh evaluation for Reacher D2P P1.
#
# This is intentionally serial: each stage consumes an immutable receipt from
# the preceding stage.  It never reuses the opened P1 roots that diagnosed the
# original frozen checkpoint, never loosens a gate, and never falls back to
# CPU.  Invoke from the host, e.g.:
#   systemd-run --user --scope --quiet --collect --setenv=CUDA_VISIBLE_DEVICES=0 \
#     bash experiments/spectral_reachability/run_reacher_d2p_causal_dynamics_repair_cuda.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${1:-${ROOT}/results/reacher_d2p_causal_dynamics_repair_20261024_cuda}"
SOURCE_DIR="${RUN_ROOT}/training_pool"
DATASET="${SOURCE_DIR}/reacher.h5"
CHECKPOINT="${RUN_ROOT}/checkpoint"
ADAPTER="${ROOT}/results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"

if [[ -e "${RUN_ROOT}" && -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite immutable repair run: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${ADAPTER}" ]]; then
  echo "missing frozen Reacher action-normalization receipt: ${ADAPTER}" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p "${RUN_ROOT}"
.venv/bin/python -m experiments.spectral_reachability.build_reacher_d2p_repair_training_pool \
  --output "${SOURCE_DIR}" \
  --device cuda \
  --seed 20264800 \
  --episodes 128 \
  --episode-steps 64

.venv/bin/python -m experiments.spectral_reachability.train_reacher_d2p_dynamics_repair \
  --dataset "${DATASET}" \
  --adapter-artifact "${ADAPTER}" \
  --output "${CHECKPOINT}" \
  --device cuda \
  --seed 20265001 \
  --epochs 12 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --ranking-temperature 0.1

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p0 \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --device cuda \
  --seed 20265100 \
  --roots 4 \
  --output "${RUN_ROOT}/p0_preflight"

for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${CHECKPOINT}" \
    --adapter-artifact "${ADAPTER}" \
    --device cuda \
    --seed 20265200 \
    --roots 4 \
    --calibration-roots 4 \
    --calibration-only \
    --damping "${damping}" \
    --output "${RUN_ROOT}/p0_calibration_d${damping}"
done

.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports \
    "${RUN_ROOT}/p0_calibration_d0.001/report.json" \
    "${RUN_ROOT}/p0_calibration_d0.01/report.json" \
    "${RUN_ROOT}/p0_calibration_d0.1/report.json" \
    "${RUN_ROOT}/p0_calibration_d1/report.json" \
    "${RUN_ROOT}/p0_calibration_d10/report.json" \
  --output "${RUN_ROOT}/p0_damping_selection.json"

DAMPING="$(jq -er '.selected.damping' "${RUN_ROOT}/p0_damping_selection.json")"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --device cuda \
  --seed 20265300 \
  --roots 12 \
  --calibration-roots 0 \
  --outcome-only \
  --damping "${DAMPING}" \
  --calibration-evidence "${RUN_ROOT}/p0_damping_selection.json" \
  --output "${RUN_ROOT}/p0_outcome"

if [[ "$(jq -r '.stage_gate.p1_unlocked' "${RUN_ROOT}/p0_outcome/report.json")" != "true" ]]; then
  echo "P0 did not unlock P1 for the repaired checkpoint; stop before P1 as required by the protocol." >&2
  exit 3
fi

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --device cuda \
  --seed 20265400 \
  --roots 4 \
  --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --output "${RUN_ROOT}/p1_preflight"

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --device cuda \
  --seed 20265500 \
  --roots 6 \
  --calibration-only \
  --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" \
  --damping "${DAMPING}" \
  --mechanism-observability observed_encoded_goal_cost \
  --output "${RUN_ROOT}/p1_calibration"

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --report "${RUN_ROOT}/p1_calibration/report.json" \
  --output "${RUN_ROOT}/p1_first_chunk_fidelity.json"

if [[ "$(jq -r '.gates.every_model_eligible_reconfiguration_is_observably_non_greedy_relative_to_zero_prefix' "${RUN_ROOT}/p1_calibration/report.json")" != "true" ]]; then
  echo "repaired checkpoint still fails the observed-successor calibration gate; outcome roots remain sealed." >&2
  exit 4
fi

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" \
  --adapter-artifact "${ADAPTER}" \
  --device cuda \
  --seed 20265600 \
  --roots 12 \
  --outcome-only \
  --calibration-evidence "${RUN_ROOT}/p1_calibration/report.json" \
  --p0-evidence "${RUN_ROOT}/p0_outcome/report.json" \
  --p1-preflight-evidence "${RUN_ROOT}/p1_preflight" \
  --damping "${DAMPING}" \
  --mechanism-observability observed_encoded_goal_cost \
  --output "${RUN_ROOT}/p1_outcome"
