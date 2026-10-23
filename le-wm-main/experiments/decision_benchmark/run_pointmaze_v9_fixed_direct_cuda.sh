#!/usr/bin/env bash
# Fresh, no-selector confirmation after the opened V8 development panel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"; WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
DATASET="${DATASET:-${ROOT}/data/point_maze_v9_extension_20261021}"
OUT="${OUT:-${ROOT}/results/pointmaze_v9_fixed_direct_confirmation_20261021_cuda}"
TRAIN_ROOTS="${TRAIN_ROOTS:-64}"; CALIBRATION_ROOTS="${CALIBRATION_ROOTS:-48}"; OUTCOME_ROOTS="${OUTCOME_ROOTS:-32}"
RUNNER="${RUNNER:-experiments.decision_benchmark.run_pointmaze_temporal_ood_v9_fixed_direct}"
ROOT_SEED="${ROOT_SEED:-202610310}"; HELD_ACTION_SEED="${HELD_ACTION_SEED:-202610311}"; TRAIN_ACTION_SEED="${TRAIN_ACTION_SEED:-202610312}"
OBSERVATION_CHART="${OBSERVATION_CHART:-latent_history}"
CHART_ARGS=()
if [[ "${RUNNER}" == *"run_pointmaze_temporal_ood_v12_decoupled_belief" ]]; then
  CHART_ARGS=(--observation-chart "${OBSERVATION_CHART}")
fi
if [[ -e "${OUT}" ]]; then echo "refusing to reuse sealed V9 output" >&2; exit 2; fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
"${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke
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
 results/pointmaze_v8_visual_belief_state_20261021_cuda/temporal_ood_v6_context.json
 results/pointmaze_v9_fixed_direct_confirmation_20261021_cuda/temporal_ood_v6_context.json
 results/pointmaze_v10_expanded_calibration_confirmation_20261021_cuda/temporal_ood_v6_context.json
 results/pointmaze_v11_fixed_belief_confirmation_20261021_cuda/temporal_ood_v6_context.json
 results/pointmaze_v12_decoupled_belief_confirmation_20261021_cuda/temporal_ood_v6_context.json
 results/pointmaze_v13_decoupled_belief_large_confirmation_20261021_cuda/temporal_ood_v6_context.json
)
"${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
 --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${OUT}" \
 --encoder-checkpoint "${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar" \
 --pca-checkpoint "${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt" \
 --dinov2-source "${ROOT}/external/local_dinov2" --candidate-generalization-regime shared_candidate_in_support \
 --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
 --root-seed "${ROOT_SEED}" --held-action-seed "${HELD_ACTION_SEED}" --train-action-seed "${TRAIN_ACTION_SEED}" \
 --train-root-count "${TRAIN_ROOTS}" --calibration-root-count "${CALIBRATION_ROOTS}" --outcome-root-count "${OUTCOME_ROOTS}" --device cuda
"${WRAPPER}" "${PYTHON}" -m "${RUNNER}" calibrate \
 --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
 --context "${OUT}/temporal_ood_v6_context.json" --calibration-labels "${OUT}/temporal_ood_v6_calibration_outcomes.json" \
 --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/fixed_direct_frozen.json" --device cuda "${CHART_ARGS[@]}"
"${WRAPPER}" "${PYTHON}" -m "${RUNNER}" score \
 --source "${OUT}/temporal_ood_v6_source_dataset.json" --source-arrays "${OUT}/temporal_ood_v6_source_arrays.npz" \
 --context "${OUT}/temporal_ood_v6_context.json" --frozen-selection "${OUT}/fixed_direct_frozen.json" \
 --manifest "${OUT}/temporal_ood_v6_manifest.json" --output "${OUT}/score" --device cuda "${CHART_ARGS[@]}"
