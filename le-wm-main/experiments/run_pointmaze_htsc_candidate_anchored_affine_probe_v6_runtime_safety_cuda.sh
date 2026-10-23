#!/usr/bin/env bash
# Fresh CUDA-only V6 confirmation of the candidate-anchored active probe.
# V5 is quarantined because an initially safe pulse became backward after
# earlier probes.  V6 rechecks task safety before every physical pulse and
# uses both a new state/action extension and V5-disjoint calibration roots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_pointmaze_htsc_active_probe_strict"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
CALIBRATION_DATASET="${ROOT}/data/point_maze_tail1800_1899"
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v30_fixed_counterfactual_confirmation_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v31_candidate_anchored_runtime_safety_extension_20261022"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_candidate_anchored_affine_probe_v6_runtime_safety_20261022_cuda"
ACTIVE_METHOD="effect_adapter_rgb_online_affine_candidate_anchored_verified_local"
EXTENSION_SEED=202610601
ROOT_MIN=12920
ROOT_MAX_EXCLUSIVE=13128

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

# V6 has never used these calibration roots.  They remain root-disjoint from
# the new renderer-on-demand held-out extension and from V5's 1800--1831
# calibration/gate roots.
VISUAL_ROOTS="1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844,1845,1846,1847"
GATE_ROOTS="1848,1849,1850,1851,1852,1853,1854,1855,1856,1857,1858,1859,1860,1861,1862,1863"
TEST_ROOTS="$(seq -s, "${ROOT_MIN}" $((ROOT_MIN + 63)))"

# The allocation preflight is authoritative; it rejects CPU fallback before
# any immutable result path is created.
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
