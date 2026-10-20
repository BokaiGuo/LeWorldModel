#!/usr/bin/env bash
set -euo pipefail

# Formal fresh-root protocol for the repaired online local-effect path.
# This script is intentionally fail-closed: the first Python process must pass
# the real CUDA allocation probe, and no CPU fallback is permitted.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${PROJECT_ROOT}/data/point_maze_tail1800_1899"
EFFECT_DATASET="${PROJECT_ROOT}/results/htsc_preprocessing_repair_20261016/runtime_pulse_bank_rank32_renderer_development/candidate_dataset.json"
EFFECT_ARRAYS="${PROJECT_ROOT}/results/htsc_preprocessing_repair_20261016/runtime_pulse_bank_rank32_renderer_development/arrays.npz"
SOURCE_CHECKPOINT="${PROJECT_ROOT}/results/htsc_preprocessing_repair_20261016/deployment_aligned_renderer_adapter_rank32/checkpoints/jepa_renderer_matched_task_effect_mlp/seed_8411/stage_005.pt"
ARTIFACT_ROOT="${PROJECT_ROOT}/results/gdd_wm_online_local_effect_20261019_cuda_fresh20"
POLICY="${ARTIFACT_ROOT}/effect_policy.pt"
SUMMARY="${ARTIFACT_ROOT}/effect_policy_summary.json"
REPORT="${ARTIFACT_ROOT}/planning_report.json"

# The effect policy is fitted on roots 1800--1853 and calibrated on 1854--1867.
# The 20 outcome roots below are unopened by this protocol and are not read by
# the fitting command.  They are kept explicit for root-disjoint auditing.
CALIBRATION_EPISODES="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
TEST_EPISODES="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887"

mkdir -p "${ARTIFACT_ROOT}"

echo "[formal] CUDA preflight: this must allocate a real CUDA tensor"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda

echo "[formal] fitting exact runtime_repeated_pulse_v1 / goal_relative policy"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.decision_benchmark.fit_pointmaze_online_effect_policy \
  --candidate-dataset "${EFFECT_DATASET}" \
  --arrays "${EFFECT_ARRAYS}" \
  --source-model-checkpoint "${SOURCE_CHECKPOINT}" \
  --output "${POLICY}" \
  --summary-output "${SUMMARY}" \
  --device cuda \
  --calibration-splits calibration

echo "[formal] running matched four-arm fresh-root panel"
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

echo "[formal] report: ${REPORT}"
