#!/usr/bin/env bash
set -euo pipefail

# Deployment-matched history-4 protocol.  The source export, learned teacher
# adapter, live frozen LeWM checkpoint, and RGB preprocessing contract all
# refer to the same released PointMaze model.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SOURCE_EXPORT="${PROJECT_ROOT}/results/protocol_corrected_temporal_export_20261012/jepa_pointmaze_windows4_100_metrics_enriched_export.json"
DEPLOYMENT_CHECKPOINT="${PROJECT_ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
ADAPTER_ROOT="${PROJECT_ROOT}/results/gdd_wm_runtime_aligned_history4_adapter_20261019"
ADAPTER="${ADAPTER_ROOT}/checkpoints/jepa_runtime_mlp/seed_8411/stage_005.pt"
ARTIFACT_ROOT="${PROJECT_ROOT}/results/gdd_wm_runtime_aligned_history4_effect_20261019_cuda_fresh20"
CANDIDATE_ROOT="${ARTIFACT_ROOT}/candidate_bank"
DATASET="${CANDIDATE_ROOT}/candidate_dataset.json"
ARRAYS="${CANDIDATE_ROOT}/arrays.npz"
POLICY="${ARTIFACT_ROOT}/effect_policy.pt"
SUMMARY="${ARTIFACT_ROOT}/effect_policy_summary.json"
REPORT="${ARTIFACT_ROOT}/planning_report.json"

CALIBRATION_EPISODES="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
TEST_EPISODES="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887"

mkdir -p "${ARTIFACT_ROOT}"

echo "[formal-history4-matched] CUDA preflight"
"${PYTHON}" -m experiments.gpu_preflight --device cuda

echo "[formal-history4-matched] fitting deployment-matched history-4 teacher adapter"
"${PYTHON}" -m experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel \
  --single-export "${SOURCE_EXPORT}" \
  --single-base-name jepa_runtime \
  --output "${ADAPTER_ROOT}" \
  --rank 8 --epochs 5 --stages 5 --architectures mlp \
  --history-windows 4 --action-feature-chart raw --seeds 8411 \
  --outcome-count 32 --device cuda

echo "[formal-history4-matched] generating exact runtime candidate bank"
rm -rf "${CANDIDATE_ROOT}"
"${PYTHON}" -m experiments.decision_benchmark.generate_candidate_bank \
  --source-export "${SOURCE_EXPORT}" \
  --checkpoint "${ADAPTER}" \
  --output "${CANDIDATE_ROOT}" \
  --goal-x 1.2 --goal-y 1.2 --candidates 32 --steps 20 --max-instances 100 \
  --device cuda --candidate-family runtime_repeated_pulse_v1 \
  --source-selection sorted_episode --split-assignment checkpoint_protocol_v1 \
  --source-window-start 15 --history-windows 4 --canonical-action-bank

echo "[formal-history4-matched] fitting train-only effect policy"
"${PYTHON}" -m experiments.decision_benchmark.fit_pointmaze_online_effect_policy \
  --candidate-dataset "${DATASET}" --arrays "${ARRAYS}" \
  --source-model-checkpoint "${ADAPTER}" --output "${POLICY}" \
  --summary-output "${SUMMARY}" --device cuda \
  --feature-mode history_latent_action_basis --calibration-splits calibration

echo "[formal-history4-matched] running fresh-root CUDA panel"
"${PYTHON}" experiments/run_live_pointmaze_planning.py \
  --repo-root "${PROJECT_ROOT}" \
  --dataset-root "${PROJECT_ROOT}/data/point_maze_tail1800_1899" \
  --checkpoint "${DEPLOYMENT_CHECKPOINT}" \
  --dinov2-source "${PROJECT_ROOT}/external/local_dinov2" \
  --effect-policy-checkpoint "${POLICY}" \
  --calibration-episodes "${CALIBRATION_EPISODES}" \
  --test-episodes "${TEST_EPISODES}" \
  --test-visual-source stored_episode --planner-map rgb_maze \
  --max-blocks 40 --effect-regret-tolerance 0.5 \
  --methods visual_dynamics effect_adapter_rgb effect_adapter_rgb_online \
    effect_adapter_rgb_online_htsc effect_adapter_rgb_online_probe \
    effect_adapter_rgb_online_verified_local effect_adapter_rgb_online_local_jacobian \
  --output "${REPORT}"

echo "[formal-history4-matched] report: ${REPORT}"
