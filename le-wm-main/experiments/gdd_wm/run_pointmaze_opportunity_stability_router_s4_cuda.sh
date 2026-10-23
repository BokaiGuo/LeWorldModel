#!/usr/bin/env bash
# Fresh host-CUDA S4: prospective calibration-power replication with a denser
# pre-outcome tail-headroom bank.  S3 was rejected statically before model
# fitting because its bank missed the unchanged headroom contract.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BASE_POOL="${ROOT}/data/point_maze_v42_opportunity_stability_s3_power_extension_20261022"
DATASET="${ROOT}/data/point_maze_v43_opportunity_stability_s4_extension_20261022"
PREFLIGHT="${ROOT}/results/pointmaze_opportunity_stability_s4_static_preflight_20261022.json"
PANEL="${ROOT}/results/pointmaze_opportunity_stability_s4_panel_20261022_cuda"
OUTPUT="${ROOT}/results/pointmaze_opportunity_stability_s4_control_20261022_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"

[[ ! -e "${PREFLIGHT}" && ! -e "${PANEL}" && ! -e "${OUTPUT}" ]] || { echo "refusing to reuse immutable S4 preflight, panel, or output path" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${BASE_POOL}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing required S4 input: ${required}" >&2; exit 3; }
done
for attempt in 1 2 3; do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then break; fi
  [[ "${attempt}" == 3 ]] && { echo "CUDA recurrent preflight failed; refusing CPU fallback" >&2; exit 9; }
  sleep 5
done
if [[ ! -e "${DATASET}" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.extend_pointmaze_state_action_pool \
    --input-root "${BASE_POOL}" --output "${DATASET}" --episode-count 640 --seed 202611300
fi
mapfile -t CONTEXTS < <(find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
  jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null && readlink -f "${path}"
done)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts available for S4 identity exclusion" >&2; exit 7; }
"${PYTHON}" -m experiments.gdd_wm.preflight_pointmaze_opportunity_router_v9 \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed 202611301 --held-action-seed 202611302 --train-action-seed 202611303 \
  --extension-seed 202611300 --root-id-min 16440 --root-id-max-exclusive 17080 \
  --candidate-design ultra_tail_robust_goal_balanced_interaction_pairs \
  --train-root-count 160 --calibration-root-count 288 --outcome-root-count 64 \
  --router-kind advantage_bootstrap_lower --output "${PREFLIGHT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support --candidate-design ultra_tail_robust_goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202611301 --held-action-seed 202611302 --train-action-seed 202611303 \
  --train-root-count 160 --calibration-root-count 288 --outcome-root-count 64 \
  --root-id-min 16440 --root-id-max-exclusive 17080 --device cuda --frozen-preflight-ledger "${PREFLIGHT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.gdd_wm.run_pointmaze_planning_shaped_repairability \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202611304 --width 32 --steps 400 --ensemble-size 4 \
  --router-kind advantage_bootstrap_lower --action-representation full_ordered_20x2 --require-shared-candidate-support --opportunity-gated
"${PYTHON}" -m experiments.gdd_wm.audit_planning_shaped_repairability_mechanism \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/planning_shaped_repairability_mechanism_audit.json"
