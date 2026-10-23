#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
RUNNER_MODULE="experiments.decision_benchmark.run_pointmaze_temporal_ood_v6_interaction"
BUILDER_MODULE="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
DATASET="${ROOT}/data/point_maze_tail1800_1899"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
# This train-only adapter checkpoint is used only as the existing PCA artifact:
# the builder reads its pca_mean/pca_basis fields and records its SHA-256.
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
OUT="${ROOT}/results/pointmaze_temporal_ood_v6_9_protocol_repair_20261021_cuda"

if [[ -e "${OUT}" ]]; then
  echo "refusing to reuse existing V6.9 output path: ${OUT}" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing CPU fallback" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
experiments/run_with_torch_cudnn.sh "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

CONTEXTS=(
  results/pointmaze_temporal_ood_20261021_cuda/temporal_ood_context.json
  results/pointmaze_temporal_ood_v2_20261021_cuda/temporal_ood_v2_context.json
  results/pointmaze_temporal_ood_v3_20261021_cuda/temporal_ood_v3_context.json
  results/pointmaze_temporal_ood_v4_20261021_cuda/temporal_ood_v4_context.json
  results/pointmaze_temporal_ood_v5_20261021_cuda/temporal_ood_v5_context.json
  results/pointmaze_temporal_ood_v6_1_matched_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_2_matched_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_3_matched_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_4_goal_sensitivity_20261021_cuda/goal_sensitivity_context.json
  results/pointmaze_temporal_ood_v6_4_in_support_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_5_observability_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_6_history_gate_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_7_state_interaction_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_8_goal_sensitivity_20261021_cuda_retry/goal_sensitivity_context.json
  results/pointmaze_temporal_ood_v6_8_state_interaction_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_temporal_ood_v6_matched_20261021_cuda/temporal_ood_v6_context.json
)

"${PYTHON}" -m "${BUILDER_MODULE}" \
  --dataset-root "${DATASET}" \
  --contexts "${CONTEXTS[@]}" \
  --output "${OUT}" \
  --encoder-checkpoint "${ENCODER}" \
  --pca-checkpoint "${PCA}" \
  --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design state_interaction_pairs \
  --goal-qpos 0.8 0.8 \
  --device cuda

experiments/run_with_torch_cudnn.sh "${PYTHON}" -m "${RUNNER_MODULE}" calibrate \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" \
  --calibration-labels "${OUT}/temporal_ood_v6_calibration_outcomes.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" \
  --output "${OUT}/temporal_ood_v6_frozen_selection.json" \
  --epochs 100 --device cuda

experiments/run_with_torch_cudnn.sh "${PYTHON}" -m "${RUNNER_MODULE}" score \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" \
  --frozen-selection "${OUT}/temporal_ood_v6_frozen_selection.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" \
  --output "${OUT}/score" \
  --device cuda
