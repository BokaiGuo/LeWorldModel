#!/usr/bin/env bash
# Fresh GPU-only V9: test whether routing improves once non-opportunity roots
# are explicitly excluded from router fitting, calibration, and release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BASE_POOL="${ROOT}/data/point_maze_v37_opportunity_router_r2_extension_20261022"
DATASET="${ROOT}/data/point_maze_v38_opportunity_router_r3_extension_20261022"
PREFLIGHT="${ROOT}/results/pointmaze_opportunity_router_v9_r3_static_preflight_20261022.json"
PANEL="${ROOT}/results/pointmaze_opportunity_router_v9_r3_panel_20261022_cuda"
OUTPUT="${ROOT}/results/pointmaze_opportunity_router_v9_r3_control_20261022_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
RETRY_SECONDS="${GPU_PREFLIGHT_RETRY_SECONDS:-5}"

if [[ -e "${PANEL}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable V9 panel or output path" >&2; exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${BASE_POOL}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing required V9 input: ${required}" >&2; exit 3; }
done
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then break; fi
  if (( attempt == ATTEMPTS )); then
    echo "CUDA recurrent preflight failed after ${ATTEMPTS} attempt(s); refusing CPU fallback" >&2; exit 9
  fi
  sleep "${RETRY_SECONDS}"
done
# The original V9 root range had been opened by a later counterfactual panel.
# This r2 experiment starts from the newest state-only pool and appends a new
# 512-root range after CUDA admission.  It consumes neither a fitted local
# repair model nor any outcome artifact from that pool's historical lineage.
if [[ ! -e "${DATASET}" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.extend_pointmaze_state_action_pool \
    --input-root "${BASE_POOL}" --output "${DATASET}" --episode-count 512 --seed 202610930
fi
mapfile -t CONTEXTS < <(find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
  if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null; then
    readlink -f "${path}"
  fi
done)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts available for V9 identity exclusion" >&2; exit 7; }
"${PYTHON}" -m experiments.gdd_wm.preflight_pointmaze_opportunity_router_v9 \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed 202610931 --held-action-seed 202610932 --train-action-seed 202610933 \
  --output "${PREFLIGHT}" --reuse-identical-output
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610931 --held-action-seed 202610932 --train-action-seed 202610933 \
  --train-root-count 160 --calibration-root-count 160 --outcome-root-count 64 \
  --root-id-min 14264 --root-id-max-exclusive 14776 --device cuda --frozen-preflight-ledger "${PREFLIGHT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.gdd_wm.run_pointmaze_planning_shaped_repairability \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610934 --width 32 --steps 400 --ensemble-size 4 \
  --router-kind advantage_ridge --action-representation full_ordered_20x2 --require-shared-candidate-support --opportunity-gated
"${PYTHON}" -m experiments.gdd_wm.audit_planning_shaped_repairability_mechanism \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/planning_shaped_repairability_mechanism_audit.json"
