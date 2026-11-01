#!/usr/bin/env bash
# Host-CUDA-only immutable P0 admission for the independent TwoRoom D2P path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-${ROOT}/results/tworoom_d2p_p0_preflight_20261025_cuda}"

if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; refusing CPU fallback." >&2
  exit 2
fi

cd "${ROOT}"
exec "${ROOT}/.venv/bin/python" -m experiments.spectral_reachability.preflight_tworoom_d2p_p0 \
  --device cuda \
  --output "${OUTPUT}" \
  --policy /home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm \
  --training-action-stats "${ROOT}/results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json" \
  --roots 4 \
  --seed 202610250 \
  --history-len 3 \
  --future-horizon 2 \
  --action-block 5 \
  --probe-dims 8 \
  --fd-pairs 8 \
  --fd-epsilon 0.001
