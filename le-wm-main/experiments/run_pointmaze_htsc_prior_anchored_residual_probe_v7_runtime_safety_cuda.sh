#!/usr/bin/env bash
# Fresh host-CUDA V7 test of a calibration-prior-anchored local residual chart.
# V6 established that a full affine intercept-plus-Jacobian model is usually
# not identifiable under safe sequential probes.  V7 predeclares calibrated
# visual dynamics as the zero-action anchor, fits only its RGB-observed action
# residual, and releases only candidates inside conv(0, safe probe pair).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_pointmaze_htsc_active_probe_strict"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
CALIBRATION_DATASET="${ROOT}/data/point_maze_tail1800_1899"
# Continue from V6's independently generated extension so the V7 additions
# begin strictly after root 13127.  V7 never evaluates inherited roots.
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v31_candidate_anchored_runtime_safety_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v32_prior_anchored_residual_extension_20261022"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_prior_anchored_residual_probe_v7_runtime_safety_20261022_cuda"
ACTIVE_METHOD="effect_adapter_rgb_online_prior_anchored_candidate_residual_verified_local"
EXTENSION_SEED=202610701
ROOT_MIN=13128
ROOT_MAX_EXCLUSIVE=13336

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable active-probe output path: ${OUTPUT}" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${SOURCE_STATE_DATASET}" "${V6_MANIFEST}" "${CHECKPOINT}" "${DINOV2}" "${EFFECT_POLICY}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 3; }
done

if [[ ! -e "${FRESH_STATE_DATASET}" ]]; then
  "${PYTHON}" -m "${EXTENDER}" \
    --input-root "${SOURCE_STATE_DATASET}" --output "${FRESH_STATE_DATASET}" \
    --episode-count $((ROOT_MAX_EXCLUSIVE - ROOT_MIN)) --seed "${EXTENSION_SEED}"
fi
MANIFEST="${FRESH_STATE_DATASET}/extension_manifest.json"
[[ -f "${MANIFEST}" ]] || { echo "missing fresh extension manifest" >&2; exit 4; }
jq -e \
  --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" \
  --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" \
  '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' \
  "${MANIFEST}" >/dev/null || { echo "fresh extension contract mismatch" >&2; exit 5; }

# These roots are disjoint from V6's 1832--1863 runtime calibration panel and
# from all V7 fresh test roots.  They freeze candidate schedule/authorization
# before held-out states are read.
VISUAL_ROOTS="1864,1865,1866,1867,1868,1869,1870,1871,1872,1873,1874,1875,1876,1877,1878,1879"
GATE_ROOTS="1880,1881,1882,1883,1884,1885,1886,1887,1888,1889,1890,1891,1892,1893,1894,1895"
TEST_ROOTS="$(seq -s, "${ROOT_MIN}" $((ROOT_MIN + 63)))"

# Codex's bwrap sandbox cannot expose NVIDIA device nodes.  This script is
# deliberately launched by the host service/direct terminal; the preflight
# rejects any CPU fallback before it creates an immutable result directory.
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

export POINTMAZE_HTSC_ACTIVE_METHOD="${ACTIVE_METHOD}"
bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --dataset-root "${CALIBRATION_DATASET}" \
  --test-state-root "${FRESH_STATE_DATASET}" \
  --test-visual-source dataset_matched_renderer \
  --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}" \
  --effect-policy-checkpoint "${EFFECT_POLICY}" --source-manifest "${V6_MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --visual-calibration-roots "${VISUAL_ROOTS}" --gate-calibration-roots "${GATE_ROOTS}" \
  --test-roots "${TEST_ROOTS}" --probe-direction-grid 2 \
  --probe-magnitude-grid 0.25,0.5 \
  --active-probe-trigger-grid conditional_state_support_failure,conditional_decision_ambiguity \
  --probe-ambiguity-margin-grid 0.01,0.03,0.05 \
  --total-block-budget 25

"${PYTHON}" -m experiments.audit_pointmaze_active_probe_trace_contract \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/trace_contract_audit.json"
"${PYTHON}" -m experiments.audit_pointmaze_active_probe_identifiability \
  --report "${OUTPUT}/report.json" --active-method "${ACTIVE_METHOD}" \
  --output "${OUTPUT}/identifiability_audit.json"
