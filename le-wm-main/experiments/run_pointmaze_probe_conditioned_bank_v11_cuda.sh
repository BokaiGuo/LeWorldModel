#!/usr/bin/env bash
# Host-CUDA V11 mechanism panel: observed RGB probe chart -> locally supported
# candidate bank -> local direct action ranking.  This is intentionally not a
# strict shared-bank scorer comparison: changing the candidate set is the
# intervention under test.  Every probe consumes the same action budget as a
# planning pulse and a failed local construction keeps the root-fixed RGB
# fallback bank.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
CALIBRATION_DATASET="${ROOT}/data/point_maze_v36_passive_local_renderer_calibration_20261022"
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v46_probe_conditioned_bank_r4_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v47_probe_conditioned_bank_r5_extension_20261022"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
OUTPUT="${ROOT}/results/pointmaze_probe_conditioned_bank_v11_r5_20261022_cuda/report.json"
EXTENSION_SEED=202611050
ROOT_MIN=17656
ROOT_MAX_EXCLUSIVE=17864
CALIBRATION_ROOTS="1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943"
TEST_ROOTS="$(seq -s, "${ROOT_MIN}" $((ROOT_MIN + 63)))"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
RETRY_SECONDS="${GPU_PREFLIGHT_RETRY_SECONDS:-5}"

[[ ! -e "${OUTPUT}" ]] || { echo "immutable V11 output already exists: ${OUTPUT}" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${SOURCE_STATE_DATASET}" "${CHECKPOINT}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing V11 input: ${required}" >&2; exit 3; }
done

for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then
    break
  fi
  if (( attempt == ATTEMPTS )); then
    echo "host CUDA recurrent preflight failed after ${ATTEMPTS} attempts; refusing CPU fallback" >&2
    exit 9
  fi
  sleep "${RETRY_SECONDS}"
done

if [[ ! -e "${FRESH_STATE_DATASET}" ]]; then
  "${PYTHON}" -m "${EXTENDER}" \
    --input-root "${SOURCE_STATE_DATASET}" --output "${FRESH_STATE_DATASET}" \
    --episode-count $((ROOT_MAX_EXCLUSIVE - ROOT_MIN)) --seed "${EXTENSION_SEED}"
fi
MANIFEST="${FRESH_STATE_DATASET}/extension_manifest.json"
[[ -f "${MANIFEST}" ]] || { echo "missing V11 state-extension manifest" >&2; exit 4; }
jq -e \
  --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" \
  --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" \
  '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' \
  "${MANIFEST}" >/dev/null || { echo "V11 state-extension contract mismatch" >&2; exit 5; }

mkdir -p "$(dirname "${OUTPUT}")"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning \
  --dataset-root "${CALIBRATION_DATASET}" --test-state-root "${FRESH_STATE_DATASET}" \
  --test-visual-source dataset_matched_renderer \
  --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}" \
  --output "${OUTPUT}" --calibration-episodes "${CALIBRATION_ROOTS}" \
  --test-episodes "${TEST_ROOTS}" --max-blocks 20 --directions 8 \
  --magnitudes 0.25 0.50 0.75 1.00 --active-probe-directions 4 \
  --active-probe-magnitude 0.50 --active-probe-trigger always \
  --planner-map rgb_maze --methods visual_dynamics \
  effect_adapter_rgb_online_probe_conditioned_bank_verified_local \
  --seed 202611011
