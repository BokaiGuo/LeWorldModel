#!/usr/bin/env bash
# Fresh, sealed confirmation for the goal-balanced candidate-bank intervention.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
RUNNER="experiments.decision_benchmark.run_pointmaze_temporal_ood_v7_task_contrast"
DATASET="${DATASET:-${ROOT}/data/point_maze_v7_extension_20261021}"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
OUT="${OUT:-${ROOT}/results/pointmaze_v7_goal_balanced_task_contrast_20261021_cuda}"
ROOT_SEED_ARG="${ROOT_SEED_ARG:-202610270}"
HELD_ACTION_SEED_ARG="${HELD_ACTION_SEED_ARG:-202610271}"
TRAIN_ACTION_SEED_ARG="${TRAIN_ACTION_SEED_ARG:-202610272}"
EXTRA_CONTEXT="${EXTRA_CONTEXT:-}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to reuse an existing sealed V7 path: ${OUT}" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

# All earlier candidate/root panels are declared before building; their roots
# and action identities are excluded by the builder.  V6.9 is included because
# it was the development panel for this intervention.
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
  results/pointmaze_temporal_ood_v6_9_protocol_repair_20261021_cuda/temporal_ood_v6_context.json
)
if [[ -n "${EXTRA_CONTEXT}" ]]; then
  CONTEXTS+=("${EXTRA_CONTEXT}")
fi

"${WRAPPER}" "${PYTHON}" -m "${BUILDER}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${OUT}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed "${ROOT_SEED_ARG}" --held-action-seed "${HELD_ACTION_SEED_ARG}" --train-action-seed "${TRAIN_ACTION_SEED_ARG}" --device cuda

"${WRAPPER}" "${PYTHON}" -m "${RUNNER}" calibrate \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" --calibration-labels "${OUT}/temporal_ood_v6_calibration_outcomes.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/task_contrast_frozen.json" \
  --epochs 300 --device cuda

"${WRAPPER}" "${PYTHON}" -m "${RUNNER}" score \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" --frozen-selection "${OUT}/task_contrast_frozen.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/score" --device cuda
