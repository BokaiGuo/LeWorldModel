#!/usr/bin/env bash
# Fresh-root confirmation of the P1 squared pair-contrast diagnostic.  The
# metric declaration is immutable and hash-bound before calibration/outcome
# terminals are materialized; all learned work is CUDA-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"

BASE_DATASET="${ROOT}/data/point_maze_v83_endpoint_v6r3_paircontrast_20261023"
DATASET="${ROOT}/data/point_maze_v84_endpoint_v6r4_paircontrast_confirmation_20261023"
PREFLIGHT_OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r4_paircontrast_static_preflight_20261023.json"
PANEL="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r4_paircontrast_panel_20261023_cuda"
OUTPUT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6r4_paircontrast_endpoint_20261023_cuda"
P1_AUDIT="${OUTPUT}/decision_spectral_paircontrast_confirmation_audit.json"
IDENTITY_AUDIT="${OUTPUT}/endpoint_execution_identity_audit_20261023.json"
STAGE_AUDIT="${ROOT}/results/decision_spectral_stage_gates_20261023_cuda_v6r4_paircontrast.json"
SPEC="${ROOT}/experiments/decision_benchmark/pointmaze_p1_paircontrast_v6r4_spec_20261023.json"
ENCODER="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
PCA="${ROOT}/results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt"
DINOV2="${ROOT}/external/local_dinov2"

ROOT_MIN=37832
ROOT_MAX_EXCLUSIVE=38472
EXTENSION_SEED=202614101
ROOT_SEED=202614102
HELD_ACTION_SEED=202614103
TRAIN_ACTION_SEED=202614104

for required in "${PYTHON}" "${WRAPPER}" "${BASE_DATASET}" "${SPEC}" "${ENCODER}" "${PCA}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 3; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
[[ ! -e "${PANEL}" && ! -e "${OUTPUT}" && ! -e "${STAGE_AUDIT}" ]] || { echo "V6R4 evidence path already exists; refusing overwrite" >&2; exit 5; }

if [[ ! -e "${DATASET}" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.extend_pointmaze_state_action_pool \
    --input-root "${BASE_DATASET}" --output "${DATASET}" --episode-count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --seed "${EXTENSION_SEED}"
fi
[[ -f "${DATASET}/extension_manifest.json" ]] || { echo "missing V6R4 extension manifest" >&2; exit 6; }
jq -e --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" \
  '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' \
  "${DATASET}/extension_manifest.json" >/dev/null || { echo "V6R4 extension manifest mismatch" >&2; exit 7; }

mapfile -t CONTEXTS < <(
  find results -type f -name '*context*.json' | sort | while IFS= read -r path; do
    jq -e '(.metadata | type == "object") and (.records | type == "array") and all(.records[]; has("start_episode_id") and has("candidate_action_hash") and has("candidate_template_fingerprint"))' "${path}" >/dev/null && readlink -f "${path}"
  done
)
(( ${#CONTEXTS[@]} > 0 )) || { echo "no valid prior contexts for identity exclusion" >&2; exit 8; }

"${PYTHON}" -m experiments.gdd_wm.preflight_pointmaze_counterfactual_endpoint_v6 \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --root-seed "${ROOT_SEED}" --held-action-seed "${HELD_ACTION_SEED}" --train-action-seed "${TRAIN_ACTION_SEED}" \
  --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 --root-id-min "${ROOT_MIN}" --root-id-max-exclusive "${ROOT_MAX_EXCLUSIVE}" --output "${PREFLIGHT_OUTPUT}"
jq -e --arg dataset "${DATASET}" '.status == "passed_static_preflight" and .dataset == $dataset and .root_counts == {"train":128,"calibration":192,"outcome":64}' "${PREFLIGHT_OUTPUT}" >/dev/null || { echo "V6R4 candidate-bank preflight did not pass" >&2; exit 9; }
mapfile -t CONTEXTS < <(jq -r '.prior_context_sha256 | keys[]' "${PREFLIGHT_OUTPUT}")

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.build_pointmaze_temporal_ood_v6_matched_panel \
  --dataset-root "${DATASET}" --contexts "${CONTEXTS[@]}" --output "${PANEL}" --encoder-checkpoint "${ENCODER}" --pca-checkpoint "${PCA}" --dinov2-source "${DINOV2}" \
  --candidate-generalization-regime shared_candidate_in_support --candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8 \
  --root-seed "${ROOT_SEED}" --held-action-seed "${HELD_ACTION_SEED}" --train-action-seed "${TRAIN_ACTION_SEED}" --train-root-count 128 --calibration-root-count 192 --outcome-root-count 64 \
  --root-id-min "${ROOT_MIN}" --root-id-max-exclusive "${ROOT_MAX_EXCLUSIVE}" --device cuda --frozen-preflight-ledger "${PREFLIGHT_OUTPUT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control \
  --source-dataset "${PANEL}/temporal_ood_v6_source_dataset.json" --source-arrays "${PANEL}/temporal_ood_v6_source_arrays.npz" --context "${PANEL}/temporal_ood_v6_context.json" \
  --calibration-labels "${PANEL}/temporal_ood_v6_calibration_outcomes.json" --output "${OUTPUT}" --device cuda --seed 202614105 --width 32 --steps 400 --ensemble-size 4 --risk-alpha 0.1 \
  --minimum-summary-collision-pairs-per-root 8 --pre-outcome-diagnostic-spec "${SPEC}"
"${PYTHON}" -m experiments.decision_benchmark.audit_pointmaze_decision_spectral_mismatch --report "${OUTPUT}/report.json" --output "${P1_AUDIT}"
"${PYTHON}" -m experiments.decision_benchmark.audit_pointmaze_endpoint_execution_identity --report "${OUTPUT}/report.json" --output "${IDENTITY_AUDIT}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.decision_benchmark.audit_remaining_decision_spectral_stages \
  --v6r2-report "${OUTPUT}/report.json" --p1-audit "${P1_AUDIT}" --p1-identity-audit "${IDENTITY_AUDIT}" --p1-prime-audit results/pointmaze_v6r3d_pre_outcome_routing_audit_20261023.json \
  --p2-report results/decision_spectral_synthetic_p2_20261023_cuda/report.json --p3-report results/spectral_repair_synthetic_p3_20261023_cuda/report.json \
  --p3-replication-report results/spectral_repair_synthetic_p3_20261023_cuda_seed202610232/report.json --p3-replication-report results/spectral_repair_synthetic_p3_20261023_cuda_seed202610233/report.json \
  --p3v2-sweep results/spectral_repair_synthetic_p3_v2_task_metric_sweep_r2_20261023_cuda/report.json --real-repair-preflight results/pointmaze_endpoint_repairability_router_v10_r6_static_preflight_20261023.json --output "${STAGE_AUDIT}"
