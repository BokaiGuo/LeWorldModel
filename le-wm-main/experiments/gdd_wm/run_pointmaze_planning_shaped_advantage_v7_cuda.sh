#!/usr/bin/env bash
# Fresh CUDA-only V7: test magnitude-aware prospective repairability routing.
#
# V5's binary router compressed all planning-shaped wins to one label.  V7
# retains the same visual-history/action scorer arms but fits a cross-fitted
# cost-regret minus planning-regret advantage router.  The root pool is a new
# independent extension, so neither V5 nor V6 outcome labels are reopened.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
RUNNER="experiments.gdd_wm.run_pointmaze_planning_shaped_repairability"
MECHANISM_AUDIT="experiments.gdd_wm.audit_planning_shaped_repairability_mechanism"
DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
STATIC_PREFLIGHT="experiments.gdd_wm.preflight_pointmaze_planning_shaped_advantage_v7"
BASE_DATASET="${ROOT}/data/point_maze_v22_counterfactual_endpoint_extension_20261021"
DATASET="${ROOT}/data/point_maze_v23_planning_shaped_advantage_extension_20261022"
PANEL="${ROOT}/results/pointmaze_planning_shaped_advantage_fresh_v7_20261022_cuda"
OUTPUT="${ROOT}/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda"
# The original 34-context ledger remains immutable.  Four subsequently
# completed counterfactual panels added valid prior identities, so the V7 run
# must use a new 38-context receipt rather than silently reusing or replacing
# the stale exclusion set.
PREFLIGHT="${ROOT}/results/pointmaze_planning_shaped_advantage_v7_static_preflight_v7_alias_contract_r2_context38_20261022.json"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"
GPU_PREFLIGHT_ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
GPU_PREFLIGHT_RETRY_SECONDS="${GPU_PREFLIGHT_RETRY_SECONDS:-5}"

if [[ -e "${PANEL}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse a V7 panel or output path" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${BASE_DATASET}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required V7 input: ${required}" >&2
    exit 3
  fi
done
# NVML telemetry is not an execution contract: managed sandboxes can hide it
# while PyTorch has a usable CUDA device.  The recurrent preflight below is
# the authoritative fail-closed GPU gate (device nodes, allocation and the
# exact cuDNN GRU kernel used by this panel).  Some managed runtimes map the
# GPU at process start asynchronously, so retry only this pure preflight a
# bounded number of times.  No CPU fallback or panel materialization is ever
# permitted when it remains unavailable.
if [[ ! "${GPU_PREFLIGHT_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${GPU_PREFLIGHT_RETRY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "GPU_PREFLIGHT_ATTEMPTS must be positive and GPU_PREFLIGHT_RETRY_SECONDS non-negative" >&2
  exit 8
fi
gpu_ready=false
for ((attempt = 1; attempt <= GPU_PREFLIGHT_ATTEMPTS; attempt++)); do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then
    gpu_ready=true
    break
  fi
  if (( attempt < GPU_PREFLIGHT_ATTEMPTS )); then
    echo "CUDA preflight attempt ${attempt}/${GPU_PREFLIGHT_ATTEMPTS} failed; retrying in ${GPU_PREFLIGHT_RETRY_SECONDS}s" >&2
    sleep "${GPU_PREFLIGHT_RETRY_SECONDS}"
  fi
done
if [[ "${gpu_ready}" != true ]]; then
  echo "CUDA recurrent preflight failed after ${GPU_PREFLIGHT_ATTEMPTS} attempt(s); refusing CPU fallback" >&2
  exit 9
fi

# This is deterministic MuJoCo data generation, not a learned-model run.  All
# encoder fitting and all scorer/router training below are CUDA-only.  A
# matching completed extension is safely reusable after an interrupted GPU
# run; a mismatched directory always fail-closes rather than being overwritten.
if [[ -e "${DATASET}" ]]; then
  if [[ ! -f "${DATASET}/extension_manifest.json" ]] || ! jq -e \
    --arg source "$(readlink -f "${BASE_DATASET}")" \
    '(.schema == "pointmaze_independent_state_action_extension.v1") and (.source_root == $source) and (.seed == 202610370) and (.new_episode_count == 512) and (.new_root_id_range == [10296, 10807])' \
    "${DATASET}/extension_manifest.json" >/dev/null; then
    echo "V7 extension path exists but does not match the frozen extension contract" >&2
    exit 6
  fi
else
  "${PYTHON}" -m "${EXTENDER}" \
    --input-root "${BASE_DATASET}" --output "${DATASET}" --episode-count 512 --seed 202610370
fi

mapfile -t CONTEXTS < <(
  find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
    if jq -e '(.metadata | type == "object") and (.metadata.candidate_family | type == "string") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null; then
      readlink -f "${path}"
    fi
  done
)
if (( ${#CONTEXTS[@]} == 0 )); then
  echo "no valid prior contexts available for V7 identity exclusion" >&2
  exit 7
fi

# Freeze a byte-identical exclusion ledger and candidate-bank contract before
# the expensive CUDA renderer/encoder build.  It is safely reusable only when
# all discovered contexts and deterministic bank contents still agree.
"${PYTHON}" -m "${STATIC_PREFLIGHT}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed 202610371 --held-action-seed 202610372 --train-action-seed 202610373 \
  --output "${PREFLIGHT}" --reuse-identical-output
if ! jq -e '
  .post_execution_aliasing_contract == {
    "representation_key": "source_history_model",
    "radius_quantile": 0.25,
    "contrast_gap_threshold": 0.05,
    "max_reported_pairs": 64
  }
' "${PREFLIGHT}" >/dev/null; then
  echo "V7 static preflight does not bind the declared full-coverage aliasing contract" >&2
  exit 10
fi

# This new extension contributes roots 10296..10807.  The goal-balanced
# candidate design consumes 64 disjoint design roots, leaving 128/192/64
# train/calibration/outcome roots in the declared range.
bash "${WRAPPER}" "${PYTHON}" -m "${BUILDER}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610371 --held-action-seed 202610372 --train-action-seed 202610373 \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min 10296 --root-id-max-exclusive 10808 --device cuda \
  --frozen-preflight-ledger "${PREFLIGHT}"

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610374 --width 32 --steps 400 --ensemble-size 4 \
  --router-kind advantage_ridge --action-representation full_ordered_20x2 --require-shared-candidate-support

"${PYTHON}" -m "${MECHANISM_AUDIT}" \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/planning_shaped_repairability_mechanism_audit.json"

# C1 is post-execution: calibration representations set its radius without
# contrast labels, and realized contrasts are materialized only after the V7
# router and threshold are frozen.  The audit cannot alter the policy.
"${PYTHON}" -m "${ALIAS_AUDIT}" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --outcome-report "${OUTPUT}/report.json" \
  --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05 \
  --max-reported-pairs 64 \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json"

"${PYTHON}" -m "${DISAGREEMENT_AUDIT}" \
  --report "${OUTPUT}/report.json" --left-arm cost --right-arm planning \
  --decision-aliasing-audit "${OUTPUT}/decision_aliasing_calibration_outcome.json" \
  --output "${OUTPUT}/cost_planning_disagreement_audit.json"
