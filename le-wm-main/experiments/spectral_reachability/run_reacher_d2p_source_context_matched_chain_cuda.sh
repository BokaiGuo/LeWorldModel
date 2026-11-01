#!/usr/bin/env bash
# Host-CUDA closure of the source-supported Reacher P0->P1 chain.
# P0 and P1 share the exact source-contiguous warmup law.  Every output is
# immutable; P1 is reached only through a fresh P0 admission receipt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN="${1:-${ROOT}/results/d2p_reacher_default_source_context_matched_qpos_chain_20261025_cuda}"
RUN="$(dirname "${RUN}")/$(basename "${RUN}")"
CHECKPOINT="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"
STATS="${ROOT}/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json"
ENCODER_READOUT="${ROOT}/results/reacher_d2p_default_frozen_latent_qpos_readout_20261025_cuda/qpos_readout.npz"
PREDICTED_READOUT="${ROOT}/results/reacher_d2p_default_frozen_rollout_qpos_readout_20261025_cuda/predicted_qpos_readout.npz"
SOURCE_DISTRIBUTION="source_locked_training_packet_support"

[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite immutable run: ${RUN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}/weights.pt" && -f "${STATS}" && -f "${ENCODER_READOUT}" && -f "${PREDICTED_READOUT}" ]] || {
  echo "default checkpoint, source-bound adapter receipt, or paired readout is missing" >&2; exit 2;
}

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
mkdir -p "${RUN}"

for damping in 0.001 0.01 0.1 1 10; do
  .venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
    --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda \
    --context-support source_contiguous --seed 2026102800 --roots 4 --calibration-roots 4 \
    --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
    --calibration-only --damping "${damping}" --output "${RUN}/p0_calibration_d${damping}"
done
.venv/bin/python -m experiments.spectral_reachability.select_reacher_d2p_damping \
  --reports "${RUN}"/p0_calibration_d*/report.json --output "${RUN}/p0_damping_selection.json"
DAMPING="$(jq -er '.selected.damping' "${RUN}/p0_damping_selection.json")"

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda \
  --context-support source_contiguous --seed 2026102900 --roots 12 --calibration-roots 0 \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --outcome-only --damping "${DAMPING}" --calibration-evidence "${RUN}/p0_damping_selection.json" \
  --output "${RUN}/p0_outcome"
jq -e '.stage_gate.p1_unlocked == true' "${RUN}/p0_outcome/report.json" >/dev/null || {
  echo "P0 did not unlock P1 under the matched source-contiguous context; stop with immutable evidence." >&2; exit 3;
}

.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103000 --roots 4 \
  --p0-evidence "${RUN}/p0_outcome/report.json" --proposal-distribution "${SOURCE_DISTRIBUTION}" \
  --damping "${DAMPING}" --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --output "${RUN}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103100 --roots 6 \
  --calibration-only --p0-evidence "${RUN}/p0_outcome/report.json" --p1-preflight-evidence "${RUN}/p1_preflight" \
  --proposal-distribution "${SOURCE_DISTRIBUTION}" --damping "${DAMPING}" \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN}/p1_calibration"
jq -e '([.gates[]] | all)' "${RUN}/p1_calibration/report.json" >/dev/null

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103200 --roots 12 \
  --outcome-only --calibration-evidence "${RUN}/p1_calibration/report.json" \
  --p0-evidence "${RUN}/p0_outcome/report.json" --p1-preflight-evidence "${RUN}/p1_preflight" \
  --proposal-distribution "${SOURCE_DISTRIBUTION}" --damping "${DAMPING}" \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN}/p1_outcome"
