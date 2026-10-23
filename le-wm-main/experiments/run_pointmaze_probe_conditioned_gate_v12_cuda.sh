#!/usr/bin/env bash
# Host-CUDA V12: calibration-frozen permit/probe/release gate for the V11
# local RGB chart.  Outcome roots remain sealed unless a non-empty local
# policy is calibration-noninferior to RGB visual fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
SELECTOR="experiments.decision_benchmark.select_pointmaze_probe_conditioned_gate_v12"
CALIBRATION_DATASET="${ROOT}/data/point_maze_v36_passive_local_renderer_calibration_20261022"
SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v48_probe_conditioned_gate_v12_extension_20261022"
FRESH_STATE_DATASET="${ROOT}/data/point_maze_v49_probe_conditioned_gate_v12_r2_extension_20261022"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
DINOV2="${ROOT}/external/local_dinov2"
OUTPUT_DIR="${ROOT}/results/pointmaze_probe_conditioned_gate_v12_r2_20261022_cuda"
EXTENSION_SEED=202611130
ROOT_MIN=18072
ROOT_MAX_EXCLUSIVE=18280
FIT_ROOTS="1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943"
CAL_ROOTS="$(seq -s, 18072 18103)"
OUTCOME_ROOTS="$(seq -s, 18104 18167)"
ATTEMPTS="${GPU_PREFLIGHT_ATTEMPTS:-3}"
RETRY_SECONDS="${GPU_PREFLIGHT_RETRY_SECONDS:-5}"
MARGINS=(0.00 0.01 0.03 0.10)
ADVANTAGES=(0.00 0.01 0.03)

[[ ! -e "${OUTPUT_DIR}" ]] || { echo "immutable V12 output already exists: ${OUTPUT_DIR}" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${SOURCE_STATE_DATASET}" "${CHECKPOINT}" "${DINOV2}"; do
  [[ -e "${required}" ]] || { echo "missing V12 input: ${required}" >&2; exit 3; }
done
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke; then break; fi
  if (( attempt == ATTEMPTS )); then
    echo "host CUDA recurrent preflight failed after ${ATTEMPTS} attempt(s); refusing CPU fallback" >&2; exit 9
  fi
  sleep "${RETRY_SECONDS}"
done
if [[ ! -e "${FRESH_STATE_DATASET}" ]]; then
  "${PYTHON}" -m "${EXTENDER}" --input-root "${SOURCE_STATE_DATASET}" \
    --output "${FRESH_STATE_DATASET}" --episode-count $((ROOT_MAX_EXCLUSIVE - ROOT_MIN)) \
    --seed "${EXTENSION_SEED}"
fi
MANIFEST="${FRESH_STATE_DATASET}/extension_manifest.json"
[[ -f "${MANIFEST}" ]] || { echo "missing V12 extension manifest" >&2; exit 4; }
jq -e --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" \
  --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" \
  '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' \
  "${MANIFEST}" >/dev/null || { echo "V12 extension contract mismatch" >&2; exit 5; }

mkdir -p "${OUTPUT_DIR}/calibration"
COMMON=(--dataset-root "${CALIBRATION_DATASET}" --test-state-root "${FRESH_STATE_DATASET}"
  --test-visual-source dataset_matched_renderer --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}"
  --calibration-episodes "${FIT_ROOTS}" --test-episodes "${CAL_ROOTS}" --max-blocks 20
  --directions 8 --magnitudes 0.25 0.50 0.75 1.00 --active-probe-directions 4
  --active-probe-magnitude 0.50 --active-probe-trigger always --planner-map rgb_maze --seed 202611121)

echo "[V12] calibration baseline: no local probe"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning \
  "${COMMON[@]}" --output "${OUTPUT_DIR}/calibration/visual.json" --methods visual_dynamics

SELECT_ARGS=(--baseline "${OUTPUT_DIR}/calibration/visual.json")
for margin in "${MARGINS[@]}"; do
  for advantage in "${ADVANTAGES[@]}"; do
    label="m${margin/./p}_a${advantage/./p}"
    report="${OUTPUT_DIR}/calibration/${label}.json"
    echo "[V12] calibration candidate ${label}"
    bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning \
      "${COMMON[@]}" --output "${report}" \
      --probe-conditioned-visual-margin "${margin}" \
      --probe-conditioned-min-advantage "${advantage}" \
      --methods effect_adapter_rgb_online_probe_conditioned_bank_verified_local
    SELECT_ARGS+=(--candidate "${label}:${report}")
  done
done
SELECTION="${OUTPUT_DIR}/frozen_calibration_gate.json"
"${PYTHON}" -m "${SELECTOR}" "${SELECT_ARGS[@]}" --output "${SELECTION}"
AUTHORIZED="$(${PYTHON} -c 'import json, sys; print(int(bool(json.load(open(sys.argv[1]))["decision"]["deployment_authorized"])))' "${SELECTION}")"
if [[ "${AUTHORIZED}" != "1" ]]; then
  echo "[V12] calibration rejected every non-empty, non-inferior local gate; outcome roots remain sealed."
  exit 0
fi
mapfile -t SELECTED < <("${PYTHON}" -c 'import json, sys; x=json.load(open(sys.argv[1]))["decision"]["selected_label"]; print(x)' "${SELECTION}")
LABEL="${SELECTED[0]}"
MARGIN="${LABEL#m}"; MARGIN="${MARGIN%%_a*}"; MARGIN="${MARGIN/p/.}"
ADVANTAGE="${LABEL##*_a}"; ADVANTAGE="${ADVANTAGE/p/.}"
echo "[V12] frozen gate ${LABEL}; opening independent outcome roots"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning \
  --dataset-root "${CALIBRATION_DATASET}" --test-state-root "${FRESH_STATE_DATASET}" \
  --test-visual-source dataset_matched_renderer --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}" \
  --calibration-episodes "${FIT_ROOTS}" --test-episodes "${OUTCOME_ROOTS}" --max-blocks 20 \
  --directions 8 --magnitudes 0.25 0.50 0.75 1.00 --active-probe-directions 4 \
  --active-probe-magnitude 0.50 --active-probe-trigger always --planner-map rgb_maze \
  --probe-conditioned-visual-margin "${MARGIN}" --probe-conditioned-min-advantage "${ADVANTAGE}" \
  --seed 202611122 --output "${OUTPUT_DIR}/outcome/report.json" \
  --methods visual_dynamics effect_adapter_rgb_online_probe_conditioned_bank_verified_local
