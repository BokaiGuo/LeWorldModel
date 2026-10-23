#!/usr/bin/env bash
# Fresh CUDA-only active-probe panel on roots never used by the V6 repair panel.
# Calibration RGB is fixed from the legacy chart; held-out trajectories come
# only from the independent V22 extension and are rendered from that chart.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_pointmaze_htsc_active_probe_strict"
CALIBRATION_DATASET="${ROOT}/data/point_maze_tail1800_1899"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v22_counterfactual_endpoint_extension_20261021"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_active_probe_fresh_v2_20261022_cuda"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable active-probe output path: ${OUTPUT}" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${FRESH_STATE_DATASET}" "${V6_MANIFEST}" "${CHECKPOINT}" "${DINOV2}" "${EFFECT_POLICY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required fresh-active-probe input: ${required}" >&2
    exit 3
  fi
done

# V6 consumed its train, calibration, outcome, and interaction-design roots.
# This deterministic subtraction materializes a never-read 64-root outcome
# panel before calibration begins; its outcomes cannot influence probe choice.
mapfile -t USED < <(jq -r '[.renderer_chart_provenance.v6_split_roots_excluded[], .interaction_design_roots[]] | unique[]' "${V6_MANIFEST}")
mapfile -t FRESH < <(seq 9656 10295 | grep -F -x -v -f <(printf '%s\n' "${USED[@]}"))
if (( ${#FRESH[@]} < 64 )); then
  echo "fewer than 64 V6-disjoint roots remain in the V22 extension" >&2
  exit 4
fi
TEST_ROOTS="$(IFS=,; echo "${FRESH[*]:0:64}")"
VISUAL_ROOTS="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
GATE_ROOTS="1816,1817,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831"

# Match the strict runner's recurrent visual dependency before it can create
# an immutable output directory.
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --dataset-root "${CALIBRATION_DATASET}" \
  --test-state-root "${FRESH_STATE_DATASET}" \
  --test-visual-source dataset_matched_renderer \
  --checkpoint "${CHECKPOINT}" \
  --dinov2-source "${DINOV2}" \
  --effect-policy-checkpoint "${EFFECT_POLICY}" \
  --source-manifest "${V6_MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --visual-calibration-roots "${VISUAL_ROOTS}" \
  --gate-calibration-roots "${GATE_ROOTS}" \
  --test-roots "${TEST_ROOTS}" \
  --probe-direction-grid 4,6 \
  --probe-magnitude-grid 0.25,0.5 \
  --total-block-budget 25

"${PYTHON}" -m experiments.audit_pointmaze_active_probe_trace_contract \
  --report "${OUTPUT}/report.json" \
  --output "${OUTPUT}/trace_contract_audit.json"
