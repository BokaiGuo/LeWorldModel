#!/usr/bin/env bash
set -euo pipefail

# Formal history-2 recovery protocol.  This runner deliberately regenerates
# the history-2 candidate bank after CUDA preflight; it never reuses the old
# artifact whose arrays were accidentally copied from history-1.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_EXPORT="${PROJECT_ROOT}/results/htsc_preprocessing_repair_20261016/pointmaze_development_1800_1899_renderer_history2_enriched_export_enriched_export.json"
SOURCE_CHECKPOINT="${PROJECT_ROOT}/results/htsc_preprocessing_repair_20261016/deployment_aligned_renderer_adapter_rank32/checkpoints/jepa_renderer_matched_task_effect_mlp/seed_8411/stage_005.pt"
DATASET_ROOT="${PROJECT_ROOT}/data/point_maze_tail1800_1899"
ARTIFACT_ROOT="${PROJECT_ROOT}/results/gdd_wm_online_local_effect_history2_20261019_cuda_fresh20"
CANDIDATE_ROOT="${ARTIFACT_ROOT}/candidate_bank"
EFFECT_DATASET="${CANDIDATE_ROOT}/candidate_dataset.json"
EFFECT_ARRAYS="${CANDIDATE_ROOT}/arrays.npz"
POLICY="${ARTIFACT_ROOT}/effect_policy.pt"
SUMMARY="${ARTIFACT_ROOT}/effect_policy_summary.json"
REPORT="${ARTIFACT_ROOT}/planning_report.json"

CALIBRATION_EPISODES="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
TEST_EPISODES="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887"

mkdir -p "${ARTIFACT_ROOT}"

echo "[formal-history2] CUDA preflight: this must allocate a real CUDA tensor"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda

echo "[formal-history2] regenerating history-2 exact runtime candidate bank"
rm -rf "${CANDIDATE_ROOT}"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.decision_benchmark.generate_candidate_bank \
  --source-export "${SOURCE_EXPORT}" \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --output "${CANDIDATE_ROOT}" \
  --goal-x 1.2 --goal-y 1.2 \
  --candidates 32 --steps 20 --max-instances 100 \
  --device cuda \
  --candidate-family runtime_repeated_pulse_v1 \
  --source-selection sorted_episode \
  --split-assignment checkpoint_protocol_v1 \
  --source-window-start 5 \
  --history-windows 2 \
  --canonical-action-bank

echo "[formal-history2] fitting train-only history-2 effect policy"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.decision_benchmark.fit_pointmaze_online_effect_policy \
  --candidate-dataset "${EFFECT_DATASET}" \
  --arrays "${EFFECT_ARRAYS}" \
  --source-model-checkpoint "${SOURCE_CHECKPOINT}" \
  --output "${POLICY}" \
  --summary-output "${SUMMARY}" \
  --device cuda \
  --feature-mode history_latent_action_basis \
  --calibration-splits calibration

echo "[formal-history2] running fresh-root four-arm panel"
"${PROJECT_ROOT}/.venv/bin/python" experiments/run_live_pointmaze_planning.py \
  --repo-root "${PROJECT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${PROJECT_ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar" \
  --dinov2-source "${PROJECT_ROOT}/external/local_dinov2" \
  --effect-policy-checkpoint "${POLICY}" \
  --calibration-episodes "${CALIBRATION_EPISODES}" \
  --test-episodes "${TEST_EPISODES}" \
  --test-visual-source stored_episode \
  --planner-map rgb_maze \
  --max-blocks 40 \
  --methods visual_dynamics effect_adapter_rgb effect_adapter_rgb_online effect_adapter_rgb_online_htsc \
  --output "${REPORT}"

echo "[formal-history2] report: ${REPORT}"
