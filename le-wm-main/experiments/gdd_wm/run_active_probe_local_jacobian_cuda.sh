#!/usr/bin/env bash
set -euo pipefail

# Fresh-root CUDA consequence for the RGB-only local action-effect recovery arm.
# This arm intentionally does not load a frozen GDD effect policy: four
# predeclared cardinal pulses are observed and fit into an episode-local affine
# action-effect map before planning.  The visual-dynamics arm is the paired
# baseline on exactly the same roots.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${PROJECT_ROOT}/data/point_maze_tail1800_1899"
CHECKPOINT="${PROJECT_ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2_SOURCE="${PROJECT_ROOT}/external/local_dinov2"
ARTIFACT_ROOT="${PROJECT_ROOT}/results/gdd_wm_pointmaze_active_probe_local_jacobian_20261020_cuda_fresh20"

CALIBRATION_EPISODES="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
TEST_EPISODES="1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879,1880,1881,1882,1883,1884,1885,1886,1887"
REPORT="${ARTIFACT_ROOT}/planning_report.json"

mkdir -p "${ARTIFACT_ROOT}"

echo "[active-probe] CUDA preflight: this must allocate a real CUDA tensor"
"${PROJECT_ROOT}/.venv/bin/python" -m experiments.gpu_preflight --device cuda

echo "[active-probe] running RGB-only local affine map against visual dynamics"
"${PROJECT_ROOT}/.venv/bin/python" experiments/run_live_pointmaze_planning.py \
  --repo-root "${PROJECT_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --dinov2-source "${DINOV2_SOURCE}" \
  --calibration-episodes "${CALIBRATION_EPISODES}" \
  --test-episodes "${TEST_EPISODES}" \
  --test-visual-source stored_episode \
  --planner-map rgb_maze \
  --max-blocks 40 \
  --active-probe-directions 8 \
  --methods visual_dynamics effect_adapter_rgb_online_local_jacobian effect_adapter_rgb_online_local_direct \
  --output "${REPORT}"

echo "[active-probe] report: ${REPORT}"
