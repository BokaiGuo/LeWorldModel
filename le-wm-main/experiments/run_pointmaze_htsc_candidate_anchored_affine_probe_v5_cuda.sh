#!/usr/bin/env bash
# Fresh CUDA-only V5 active-probe panel.  It changes only V4's information
# action design: every probe is an exact member of the frozen planning bank.
# No CPU fallback and no existing result directory may be reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_pointmaze_htsc_active_probe_strict"
CALIBRATION_DATASET="${ROOT}/data/point_maze_tail1800_1899"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v22_counterfactual_endpoint_extension_20261021"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
V4_PROVENANCE="${ROOT}/results/pointmaze_htsc_affine_probe_v4_20261022_cuda/root_provenance.json"
# These roots were read by the CPU-only geometry preflight.  They are not
# eligible for the subsequent CUDA efficacy panel, even though no planning
# outcome was read, because their candidate-side geometry is now known.
PREFLIGHT_OBSERVED_ROOTS="9658,9660,9662,9664,9667,9672,9673,9679,9681,9684,9691,9697,9698,9706,9710,9712,9719,9722,9723,9725,9729,9730,9733,9734,9742,9745,9746,9756,9757,9761,9764,9768,9769,9776,9779,9781,9783,9787,9793,9797,9798,9799,9800,9802,9805,9806,9807,9809,9811,9814,9819,9822,9823,9824,9831,9832,9835,9836,9841,9846,9852,9853,9865,9866"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_candidate_anchored_affine_probe_v5_20261022_cuda"
ACTIVE_METHOD="effect_adapter_rgb_online_affine_candidate_anchored_verified_local"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable active-probe output path: ${OUTPUT}" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${FRESH_STATE_DATASET}" "${V6_MANIFEST}" "${CHECKPOINT}" "${DINOV2}" "${EFFECT_POLICY}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 3; }
done

mapfile -t USED < <(jq -r '[.renderer_chart_provenance.v6_split_roots_excluded[], .interaction_design_roots[]] | unique[]' "${V6_MANIFEST}")
IFS=',' read -r -a PREFLIGHT_ROOTS <<< "${PREFLIGHT_OBSERVED_ROOTS}"
USED+=("${PREFLIGHT_ROOTS[@]}")
ADDITIONAL_EXCLUDED_ROOTS=("${PREFLIGHT_ROOTS[@]}")
if [[ -f "${V4_PROVENANCE}" ]]; then
  mapfile -t V4_HELD < <(jq -r '.current_split.held_out_test_roots[]' "${V4_PROVENANCE}")
  USED+=("${V4_HELD[@]}")
  ADDITIONAL_EXCLUDED_ROOTS+=("${V4_HELD[@]}")
fi
ADDITIONAL_EXCLUDED_ARGS=(--additional-excluded-roots "$(IFS=,; echo "${ADDITIONAL_EXCLUDED_ROOTS[*]}")")
mapfile -t FRESH < <(seq 9656 10295 | grep -F -x -v -f <(printf '%s\n' "${USED[@]}"))
(( ${#FRESH[@]} >= 64 )) || { echo "need 64 never-used V22 roots" >&2; exit 4; }
TEST_ROOTS="$(IFS=,; echo "${FRESH[*]:0:64}")"
VISUAL_ROOTS="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
GATE_ROOTS="1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831"

# NVML can be unavailable even when the PyTorch CUDA runtime can create a
# working context (for example under a managed device namespace).  The
# allocation preflight below is the authoritative fail-closed GPU check: it
# verifies device nodes, CUDA initialization, device lookup, and an actual
# tensor allocation.  Do not reject an otherwise valid CUDA process merely
# because the optional nvidia-smi telemetry query fails.
# The inner strict runner also checks the recurrent CUDA path, but it creates
# its immutable output directory first.  Probe V5 loads the same visual
# recurrent stack, so catch a cuDNN shadowing failure here before any result
# path or root-provenance artifact exists; otherwise a failed launch can make
# the immutable path look consumed and block a lawful retry.
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
  --test-roots "${TEST_ROOTS}" --probe-direction-grid 4,6 \
  "${ADDITIONAL_EXCLUDED_ARGS[@]}" \
  --probe-magnitude-grid 0.25,0.5 \
  --active-probe-trigger-grid conditional_state_support_failure,conditional_decision_ambiguity \
  --probe-ambiguity-margin-grid 0.01,0.03,0.05 \
  --local-chart-loo-tolerance-grid 0.05,0.10,0.20 \
  --total-block-budget 25

"${PYTHON}" -m experiments.audit_pointmaze_active_probe_trace_contract \
  --report "${OUTPUT}/report.json" --output "${OUTPUT}/trace_contract_audit.json"
"${PYTHON}" -m experiments.audit_pointmaze_active_probe_identifiability \
  --report "${OUTPUT}/report.json" --active-method "${ACTIVE_METHOD}" \
  --output "${OUTPUT}/identifiability_audit.json"
