#!/usr/bin/env bash
# Fresh CUDA-only successor to the invalidated endpoint V6 panel.
#
# This launcher never reads the old V6 endpoint outcomes as an input.  It uses
# new trajectories, new root identities, a new action bank, and the repaired
# calibration-role identity binding.  The one-time trajectory extension is
# data construction only; every encoder/model fit and endpoint evaluation is
# dispatched through the CUDA wrapper below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
PREFLIGHT="experiments.gdd_wm.preflight_pointmaze_counterfactual_endpoint_v6"
BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
RUNNER="experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control"
AUDIT="experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"

BASE_DATASET="${ROOT}/data/point_maze_v80_c3_clustered_high_20261023"
DATASET="${ROOT}/data/point_maze_v81_endpoint_v6r1_extension_20261023"
EXTENSION_MANIFEST="${DATASET}/extension_manifest.json"
PREFLIGHT_OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_static_preflight_20261023.json"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_fresh_20261023_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r1_fresh_endpoint_20261023_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"

ROOT_MIN=35912
ROOT_MAX_EXCLUSIVE=36552
EXTENSION_SEED=202612001
ROOT_SEED=202612002
HELD_ACTION_SEED=202612003
TRAIN_ACTION_SEED=202612004

for required in "${PYTHON}" "${WRAPPER}" "${BASE_DATASET}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 3; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
[[ ! -e "${PREFLIGHT_OUTPUT}" && ! -e "${PANEL}" && ! -e "${OUTPUT}" ]] || {
  echo "fresh V6R1 paths already exist; refusing to overwrite evidence" >&2; exit 5;
}

# This invokes only MuJoCo trajectory construction; it is not a learned-model
# fit or evaluator and has no CPU/GPU fallback branch.
if [[ ! -e "${DATASET}" ]]; then
  "${PYTHON}" -m "${EXTENDER}" --input-root "${BASE_DATASET}" --output "${DATASET}" \
    --episode-count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --seed "${EXTENSION_SEED}"
fi
[[ -f "${EXTENSION_MANIFEST}" ]] || { echo "missing endpoint V6R1 extension manifest" >&2; exit 6; }
jq -e --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" \
  --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" \
  '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' \
  "${EXTENSION_MANIFEST}" >/dev/null || { echo "V6R1 extension manifest mismatch" >&2; exit 7; }

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda

mapfile -t CONTEXTS < <(
  find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
    if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null; then
      readlink -f "${path}"
    fi
  done
)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts for identity exclusion" >&2; exit 8; }

"${PYTHON}" -m "${PREFLIGHT}" --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed "${ROOT_SEED}" --held-action-seed "${HELD_ACTION_SEED}" --train-action-seed "${TRAIN_ACTION_SEED}" \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min "${ROOT_MIN}" --root-id-max-exclusive "${ROOT_MAX_EXCLUSIVE}" --output "${PREFLIGHT_OUTPUT}"

bash "${WRAPPER}" "${PYTHON}" -m "${BUILDER}" --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed "${ROOT_SEED}" --held-action-seed "${HELD_ACTION_SEED}" --train-action-seed "${TRAIN_ACTION_SEED}" \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min "${ROOT_MIN}" --root-id-max-exclusive "${ROOT_MAX_EXCLUSIVE}" --device cuda --frozen-preflight-ledger "${PREFLIGHT_OUTPUT}"

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202612005 --width 32 --steps 400 --ensemble-size 4 --risk-alpha 0.1 \
  --minimum-summary-collision-pairs-per-root 8

"${PYTHON}" -m "${AUDIT}" --report "${OUTPUT}/report.json" --output "${OUTPUT}/contract_audit.json"
bash "${WRAPPER}" "${PYTHON}" -m "${ALIAS_AUDIT}" --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" --outcome-report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05
"${PYTHON}" -m "${DISAGREEMENT_AUDIT}" --report "${OUTPUT}/report.json" --left-arm cost --right-arm endpoint \
  --output "${OUTPUT}/cost_endpoint_disagreement_audit.json"
