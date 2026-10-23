#!/usr/bin/env bash
# Build and execute the fresh V5 CDR-WM PointMaze control on CUDA only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
ROUTER="experiments.gdd_wm.run_pointmaze_planning_shaped_repairability"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
DATASET="${ROOT}/data/point_maze_v21_planwam_certified_repairability_extension_20261021"
PANEL="${ROOT}/results/pointmaze_planwam_certified_repairability_fresh_v5_20261021_cuda"
OUTPUT="${ROOT}/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"

if [[ -e "${PANEL}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse a V5 panel or execution path" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${DATASET}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required V5 input: ${required}" >&2
    exit 3
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing CPU fallback" >&2
  exit 4
fi

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda

mapfile -t CONTEXTS < <(
  find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
    if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null; then
      printf '%s\n' "${path}"
    fi
  done
)
if (( ${#CONTEXTS[@]} == 0 )); then
  echo "no valid prior contexts available for the V5 identity audit" >&2
  exit 5
fi

bash "${WRAPPER}" "${PYTHON}" -m "${BUILDER}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design state_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610322 --held-action-seed 202610323 --train-action-seed 202610324 \
  --root-id-min 9336 --root-id-max-exclusive 9656 --device cuda

bash "${WRAPPER}" "${PYTHON}" -m "${ROUTER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610320 --width 32 --steps 400 --ensemble-size 4

bash "${WRAPPER}" "${PYTHON}" -m "${ALIAS_AUDIT}" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --outcome-report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" \
  --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05
