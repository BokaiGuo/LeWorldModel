#!/usr/bin/env bash
# Fresh host-CUDA V8 test of a calibrated contrast release for the V7 chart.
# V7 showed that a rank-two prior-anchored residual model can be identified,
# but the inherited visual-consistency rule suppressed nearly every correction.
# V8 changes only that release gate: a correction needs a calibration-frozen
# predicted terminal-distance advantage, fixed-bank anchor support, and a
# post-action RGB successor verification.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_pointmaze_htsc_active_probe_strict"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
REFERENCE_RGB_DATASET="${ROOT}/data/point_maze_tail1800_1899"
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v32_prior_anchored_residual_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v33_prior_anchored_contrast_extension_20261022"
CALIBRATION_DATASET="${ROOT}/data/point_maze_v34_renderer_calibration_extension_20261022"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_prior_anchored_contrast_probe_v8_runtime_safety_20261022_cuda"
ACTIVE_METHOD="effect_adapter_rgb_online_prior_anchored_candidate_residual_contrast_verified_local"
EXTENSION_SEED=202610801
ROOT_MIN=13336
ROOT_MAX_EXCLUSIVE=13544
# New, V7-disjoint calibration roots.  The original archive provides RGB only
# through 1899, so their frames are materialized from the fixed renderer over
# these fresh state roots before fitting any V8 policy/calibration object.
VISUAL_ROOTS="1896,1897,1898,1899,1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,1910,1911"
GATE_ROOTS="1912,1913,1914,1915,1916,1917,1918,1919,1920,1921,1922,1923,1924,1925,1926,1927"
RENDER_REFERENCE_ROOTS="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable active-probe output path: ${OUTPUT}" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${REFERENCE_RGB_DATASET}" "${SOURCE_STATE_DATASET}" "${V6_MANIFEST}" "${CHECKPOINT}" "${DINOV2}" "${EFFECT_POLICY}"; do
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

if [[ ! -e "${CALIBRATION_DATASET}" ]]; then
  "${PYTHON}" -m experiments.decision_benchmark.materialize_pointmaze_rgb_calibration_extension \
    --source-root "${FRESH_STATE_DATASET}" --reference-root "${REFERENCE_RGB_DATASET}" \
    --output "${CALIBRATION_DATASET}" \
    --render-roots "${VISUAL_ROOTS},${GATE_ROOTS}" \
    --reference-roots "${RENDER_REFERENCE_ROOTS}"
fi
[[ -f "${CALIBRATION_DATASET}/rgb_calibration_manifest.json" ]] || {
  echo "missing renderer RGB calibration manifest" >&2; exit 6;
}
TEST_ROOTS="$(seq -s, "${ROOT_MIN}" $((ROOT_MIN + 63)))"

# Must be launched on the host: Codex's bwrap user namespace intentionally
# cannot access NVIDIA devices. The preflight rejects CPU fallback before any
# immutable output is created.
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
  --probe-magnitude-grid 0.5 \
  --active-probe-trigger-grid conditional_decision_ambiguity \
  --probe-ambiguity-margin-grid 0.01,0.03,0.05 \
  --local-contrast-min-advantage-grid 0.00,0.02,0.05 \
  --total-block-budget 25

"${PYTHON}" -m experiments.audit_pointmaze_active_probe_trace_contract \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/trace_contract_audit.json"
"${PYTHON}" -m experiments.audit_pointmaze_active_probe_identifiability \
  --report "${OUTPUT}/report.json" --active-method "${ACTIVE_METHOD}" \
  --output "${OUTPUT}/identifiability_audit.json"
