#!/usr/bin/env bash
# Host-CUDA-only immutable P1 prefix timing/fidelity audit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT}/results/tworoom_d2p_p1_prefix_fidelity_20261025_cuda}"

if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback." >&2
  exit 2
fi

cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.audit_tworoom_d2p_p1_prefix_fidelity \
  --device cuda \
  --output "${OUTPUT}" \
  --p1-evidence "${ROOT}/results/tworoom_d2p_p1_preflight_v2_20261025_cuda" \
  --calibration "${ROOT}/results/tworoom_d2p_p1_calibration_v3_20261025_cuda" \
  --policy /home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm \
  --training-action-stats "${ROOT}/results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json"
