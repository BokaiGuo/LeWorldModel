#!/usr/bin/env bash
# Fresh host-CUDA V9 test: passive prior-anchored residual repair.
#
# V6--V8 established that dedicated safe probes consume action budget and
# rarely produce enough supported candidate actions to change a decision.
# This panel therefore charges no information actions: ordinary HTSC/visual
# decisions supply the RGB-observed action-effect pairs.  A residual chart can
# re-rank only after two independent executed actions identify it, inside the
# anchor hull, with successor verification after every local release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
RUNNER="experiments.run_pointmaze_htsc_passive_local_strict"
REFERENCE_RGB_DATASET="${ROOT}/data/point_maze_tail1800_1899"
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v33_prior_anchored_contrast_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v35_passive_local_extension_20261022"
CALIBRATION_DATASET="${ROOT}/data/point_maze_v36_passive_local_renderer_calibration_20261022"
V6_MANIFEST="${ROOT}/results/pointmaze_counterfactual_endpoint_fresh_v6_20261021_cuda/temporal_ood_v6_manifest.json"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
EFFECT_POLICY="${ROOT}/results/gdd_wm_pointmaze_one_block_effect_20261020_cuda/effect_policy_live_rgb_latent.pt"
OUTPUT="${ROOT}/results/pointmaze_htsc_passive_local_v9_20261022_cuda"

# 13,544 is the first root never consumed by V5--V8.  Rendering roots are
# separately disjoint from the original archive, V8 calibrations and tests.
EXTENSION_SEED=202610901
ROOT_MIN=13544
ROOT_MAX_EXCLUSIVE=13752
VISUAL_ROOTS="1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943"
CONSISTENCY_ROOTS="1944,1945,1946,1947,1948,1949,1950,1951,1952,1953,1954,1955,1956,1957,1958,1959"
RENDER_REFERENCE_ROOTS="1800,1801,1802,1803,1804,1805,1806,1807,1808,1809,1810,1811,1812,1813,1814,1815"
TEST_ROOTS="$(seq -s, "${ROOT_MIN}" $((ROOT_MIN + 63)))"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to reuse immutable passive-local output path: ${OUTPUT}" >&2
  exit 2
fi
for required in "${PYTHON}" "${WRAPPER}" "${REFERENCE_RGB_DATASET}" "${SOURCE_STATE_DATASET}" "${V6_MANIFEST}" "${CHECKPOINT}" "${DINOV2}" "${EFFECT_POLICY}"; do
  [[ -e "${required}" ]] || { echo "missing input: ${required}" >&2; exit 3; }
done

# This script is intentionally for a host terminal/systemd user service.  A
# Codex bwrap user namespace cannot access /dev/nvidia*, and the preflight
# below rejects any CPU fallback before creating a scientific result artifact.
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

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
    --render-roots "${VISUAL_ROOTS},${CONSISTENCY_ROOTS}" \
    --reference-roots "${RENDER_REFERENCE_ROOTS}"
fi
[[ -f "${CALIBRATION_DATASET}/rgb_calibration_manifest.json" ]] || {
  echo "missing passive-local renderer RGB calibration manifest" >&2; exit 6;
}

bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" \
  --dataset-root "${CALIBRATION_DATASET}" \
  --test-state-root "${FRESH_STATE_DATASET}" \
  --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}" \
  --effect-policy-checkpoint "${EFFECT_POLICY}" --source-manifest "${V6_MANIFEST}" \
  --output-dir "${OUTPUT}" \
  --visual-calibration-roots "${VISUAL_ROOTS}" \
  --consistency-calibration-roots "${CONSISTENCY_ROOTS}" \
  --test-roots "${TEST_ROOTS}"
