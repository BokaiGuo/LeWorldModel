#!/usr/bin/env bash
# Fresh sealed visual-belief intervention.  Never reuses roots opened by V1-V7.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
DATASET="${DATASET:-${ROOT}/data/point_maze_v7_extension_20261021_b}"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
OUT="${OUT:-${ROOT}/results/pointmaze_v8_visual_belief_state_20261021_cuda}"

if [[ -e "${OUT}" ]]; then
  echo "refusing to reuse a sealed V8 output: ${OUT}" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

# These panels have all been inspected or scored.  They are declared before
# construction so neither roots nor candidate identities enter V8.
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
  results/pointmaze_v7_goal_balanced_task_contrast_20261021_cuda/temporal_ood_v6_context.json
  results/pointmaze_v7_goal_balanced_task_contrast_replication_20261021_cuda/temporal_ood_v6_context.json
)

"${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${OUT}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610300 --held-action-seed 202610301 --train-action-seed 202610302 --device cuda

"${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_temporal_ood_v8_belief_state calibrate \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" --calibration-labels "${OUT}/temporal_ood_v6_calibration_outcomes.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/belief_state_frozen.json" --epochs 500 --device cuda

"${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.run_pointmaze_temporal_ood_v8_belief_state score \
  --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
  --context "${OUT}/temporal_ood_v6_context.json" --frozen-selection "${OUT}/belief_state_frozen.json" \
  --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/score" --device cuda
