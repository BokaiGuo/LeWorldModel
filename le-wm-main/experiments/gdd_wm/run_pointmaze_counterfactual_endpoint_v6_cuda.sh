#!/usr/bin/env bash
# Fresh CUDA-only V6: test a terminal-endpoint-coupled counterfactual repair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
BUILDER="experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel"
RUNNER="experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control"
STATIC_PREFLIGHT="experiments.gdd_wm.preflight_pointmaze_counterfactual_endpoint_v6"
CONTRACT_AUDIT="experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6"
ALIAS_AUDIT="experiments.decision_benchmark.run_pointmaze_decision_aliasing_audit"
DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"
DATASET="${ROOT}/data/point_maze_v22_counterfactual_endpoint_extension_20261021"
FROZEN_PREFLIGHT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6_static_preflight_20261021.json"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_endpoint_20261021_cuda"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"

if [[ -e "${PANEL}" || -e "${OUTPUT}" ]]; then
  echo "refusing to reuse a V6 panel or endpoint-control output path" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${DATASET}" "${ENCODER}" "${PCA}" "${DINOV2}" "${FROZEN_PREFLIGHT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required V6 input: ${required}" >&2
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
      readlink -f "${path}"
    fi
  done
)
if (( ${#CONTEXTS[@]} == 0 )); then
  echo "no valid prior contexts available for V6 identity exclusion" >&2
  exit 5
fi
mapfile -t FROZEN_CONTEXTS < <(jq -r '.prior_context_sha256 | keys[]' "${FROZEN_PREFLIGHT}")
if (( ${#FROZEN_CONTEXTS[@]} == 0 )) || [[ "$(printf '%s\n' "${CONTEXTS[@]}")" != "$(printf '%s\n' "${FROZEN_CONTEXTS[@]}")" ]]; then
  echo "live prior-context discovery differs from the frozen V6 preflight ledger; refusing a non-reproducible launch" >&2
  exit 6
fi
for context in "${CONTEXTS[@]}"; do
  expected="$(jq -r --arg path "${context}" '.prior_context_sha256[$path] // empty' "${FROZEN_PREFLIGHT}")"
  actual="$(sha256sum "${context}")"; actual="${actual%% *}"
  if [[ -z "${expected}" || "${actual}" != "${expected}" ]]; then
    echo "prior context bytes differ from the frozen V6 preflight ledger: ${context}" >&2
    exit 7
  fi
done

"${PYTHON}" -m "${STATIC_PREFLIGHT}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" \
  --root-seed 202610342 --held-action-seed 202610343 --train-action-seed 202610344 \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min 9656 --root-id-max-exclusive 10296

# These counts are fixed before any V6 endpoint outcome exists.  Calibration
# roots are hash-partitioned into selector, arm-certificate, and independent
# contrast-risk roles; each arm-selection role has a support gate of 12
# model-disagreement roots.
bash "${WRAPPER}" "${PYTHON}" -m "${BUILDER}" \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" \
  --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support \
  --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed 202610342 --held-action-seed 202610343 --train-action-seed 202610344 \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min 9656 --root-id-max-exclusive 10296 --device cuda \
  --frozen-preflight-ledger "${FROZEN_PREFLIGHT}"

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" \
  --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --output "${OUTPUT}" --device cuda --seed 202610340 --width 32 --steps 400 --ensemble-size 4 --risk-alpha 0.1 \
  --minimum-summary-collision-pairs-per-root 8

"${PYTHON}" -m "${CONTRACT_AUDIT}" \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/contract_audit.json"

bash "${WRAPPER}" "${PYTHON}" -m "${ALIAS_AUDIT}" \
  --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" \
  --outcome-report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/decision_aliasing_calibration_outcome.json" \
  --representation-key source_history_model --radius-quantile 0.25 --contrast-gap-threshold 0.05

"${PYTHON}" -m "${DISAGREEMENT_AUDIT}" \
  --report "${OUTPUT}/report.json" --left-arm cost --right-arm endpoint \
  --output "${OUTPUT}/cost_endpoint_disagreement_audit.json"
