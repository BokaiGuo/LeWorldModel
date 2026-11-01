#!/usr/bin/env bash
# Independent high-power confirmation after a corrected, source-contiguous,
# readout-bound P0 chain.  No calibration or prior outcome root is reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BASE="${1:-${ROOT}/results/d2p_reacher_default_source_context_matched_qpos_chain_20261025_cuda}"
RUN="${2:-${ROOT}/results/d2p_reacher_default_source_context_matched_qpos_confirmation_n48_20261025_cuda}"
CHECKPOINT="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"
STATS="${ROOT}/results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json"
ENCODER_READOUT="${ROOT}/results/reacher_d2p_default_frozen_latent_qpos_readout_20261025_cuda/qpos_readout.npz"
PREDICTED_READOUT="${ROOT}/results/reacher_d2p_default_frozen_rollout_qpos_readout_20261025_cuda/predicted_qpos_readout.npz"
SOURCE_DISTRIBUTION="source_locked_training_packet_support"

[[ ! -e "${RUN}" && -f "${BASE}/p0_damping_selection.json" ]] || { echo "missing fixed calibration receipt or immutable output exists" >&2; exit 2; }
DAMPING="$(jq -er '.selected.damping' "${BASE}/p0_damping_selection.json")"
mkdir -p "${RUN}"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader

.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --context-support source_contiguous \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --seed 2026103300 --roots 48 --calibration-roots 0 --outcome-only --damping "${DAMPING}" \
  --calibration-evidence "${BASE}/p0_damping_selection.json" --output "${RUN}/p0_outcome"
jq -e '.stage_gate.p1_unlocked == true' "${RUN}/p0_outcome/report.json" >/dev/null || {
  echo "independent corrected P0 confirmation did not unlock P1" >&2; exit 3;
}
.venv/bin/python -m experiments.spectral_reachability.preflight_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103400 --roots 4 \
  --p0-evidence "${RUN}/p0_outcome/report.json" --proposal-distribution "${SOURCE_DISTRIBUTION}" --damping "${DAMPING}" \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" --output "${RUN}/p1_preflight"
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103500 --roots 6 --calibration-only \
  --p0-evidence "${RUN}/p0_outcome/report.json" --p1-preflight-evidence "${RUN}/p1_preflight" \
  --proposal-distribution "${SOURCE_DISTRIBUTION}" --damping "${DAMPING}" \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN}/p1_calibration"
jq -e '([.gates[]] | all)' "${RUN}/p1_calibration/report.json" >/dev/null || { echo "P1 calibration failed" >&2; exit 4; }
.venv/bin/python -m experiments.spectral_reachability.run_reacher_d2p_p1_reconfiguration \
  --policy "${CHECKPOINT}" --adapter-artifact "${STATS}" --device cuda --seed 2026103600 --roots 12 --outcome-only \
  --calibration-evidence "${RUN}/p1_calibration/report.json" --p0-evidence "${RUN}/p0_outcome/report.json" \
  --p1-preflight-evidence "${RUN}/p1_preflight" --proposal-distribution "${SOURCE_DISTRIBUTION}" --damping "${DAMPING}" \
  --qpos-readout "${ENCODER_READOUT}" --predicted-qpos-readout "${PREDICTED_READOUT}" \
  --mechanism-observability observed_encoded_goal_cost --output "${RUN}/p1_outcome"
