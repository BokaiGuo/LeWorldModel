#!/usr/bin/env bash
# Fresh GPU-only V10: select an executed endpoint-coupled repair only where
# its candidate choice differs from the cost fallback and passes calibration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BASE_POOL="${ROOT}/data/point_maze_v41_endpoint_repairability_r4_extension_20261022"
DATASET="${ROOT}/data/point_maze_v42_endpoint_repairability_r5_extension_20261022"
PREFLIGHT="${ROOT}/results/pointmaze_endpoint_repairability_router_v10_r5_static_preflight_20261022.json"
PANEL="${ROOT}/results/pointmaze_endpoint_repairability_router_v10_r5_panel_20261022_cuda"
OUTPUT="${ROOT}/results/pointmaze_endpoint_repairability_router_v10_r5_control_20261022_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
RETRY_SECONDS="${GPU_PREFLIGHT_RETRY_SECONDS:-5}"

if [[ -e "${PANEL}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable V10 panel or output path" >&2; exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${BASE_POOL}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing required V10 input: ${required}" >&2; exit 3; }
done
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then break; fi
  if (( attempt == ATTEMPTS )); then
    echo "CUDA recurrent preflight failed after ${ATTEMPTS} attempt(s); refusing CPU fallback" >&2; exit 9
  fi
  sleep "${RETRY_SECONDS}"
done
# Recreate the missing, deterministic V8/V9 state-only ancestry only after
# CUDA admission.  This is data construction, not a CPU substitute for the
# learned run: V10 reads no V9 fitted model, panel, calibration label, or
# outcome.  The two seeds/ranges are those frozen by V9's source contract.
# V10 r1 overlapped an existing context; r2/r3/r4 each failed the immutable
# q10 gap gate before fitting.  r5 adds only pre-outcome proposal coverage
# (384 state-interaction pairs) and extends r4's state-only pool after CUDA
# admission; no fitted model, calibration label, or outcome label enters the
# new root range.
if [[ ! -e "${DATASET}" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.extend_pointmaze_state_action_pool \
    --input-root "${BASE_POOL}" --output "${DATASET}" --episode-count 512 --seed 202610970
fi
mapfile -t CONTEXTS < <(find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
  if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null; then
    readlink -f "${path}"
  fi
done)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts available for V10 identity exclusion" >&2; exit 7; }
"${PYTHON}" -m experiments.gdd_wm.preflight_pointmaze_endpoint_repairability_router_v10 \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed 202610971 --held-action-seed 202610972 --train-action-seed 202610973 \
  --output "${PREFLIGHT}" --reuse-identical-output
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support --candidate-design expanded_tail_robust_goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610971 --held-action-seed 202610972 --train-action-seed 202610973 \
  --train-root-count 192 --calibration-root-count 128 --outcome-root-count 64 \
  --root-id-min 16312 --root-id-max-exclusive 16824 --device cuda --frozen-preflight-ledger "${PREFLIGHT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.gdd_wm.run_pointmaze_endpoint_repairability_router_v10 \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610974 --width 32 --steps 400 --ensemble-size 4
STATUS="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "${OUTPUT}/report.json")"
if [[ "${STATUS}" != "completed_execution_grounded_endpoint_repairability_router" ]]; then
  echo "V10 stopped before outcome materialization: ${STATUS}" >&2
  exit 0
fi
"${PYTHON}" -m experiments.gdd_wm.audit_pointmaze_endpoint_repairability_router_v10 \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/outcome_audit.json" --seed 202610975
