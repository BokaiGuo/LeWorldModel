#!/usr/bin/env bash
# CUDA-only P0 damping calibration after the Reacher visible-goal-marker fix.
# This creates calibration receipts only; it cannot open an efficacy outcome.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

POLICY="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"
STATS="results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json"
RUN="results/d2p_reacher_p0_goalmarker_damping_calibration_20261025_cuda"

for DAMPING in 0.001 0.01 0.1 1.0 10.0; do
  SAFE_DAMPING="${DAMPING//./_}"
  .venv/bin/python experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py \
    --policy "${POLICY}" --adapter-artifact "${STATS}" \
    --output "${RUN}/damping_${SAFE_DAMPING}" \
    --calibration-only --roots 4 --calibration-roots 4 --seed 2026102620 \
    --history-len 3 --future-horizon 2 --action-block 5 --damping "${DAMPING}" \
    --relinearization-steps 4 --d2p-initialization screened_gaussian_proposals \
    --d2p-proposal-samples 128 --d2p-proposal-elites 4 --d2p-restarts 1 \
    --cem-samples 32 --cem-steps 4 --cem-elite 4 --device cuda
done
