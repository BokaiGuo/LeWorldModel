#!/usr/bin/env bash
# Host-CUDA V15: the V14 residual chart was identifiable but its 0.5-radius
# probe hull could not compete with a planner whose legal actions reach 1.0.
# This is a fresh, three-way-split support intervention: same chart and gate,
# only the task-safe probe basis radius changes to 1.0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"; WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
EXTENDER="experiments.decision_benchmark.extend_pointmaze_state_action_pool"
FREEZER="experiments.decision_benchmark.freeze_pointmaze_probe_margin"
SELECTOR="experiments.decision_benchmark.select_pointmaze_probe_conditioned_gate_v12"
CALIBRATION_DATASET="${ROOT}/data/point_maze_v36_passive_local_renderer_calibration_20261022"
SOURCE_STATE_DATASET="${V15_SOURCE_STATE_DATASET:-${ROOT}/data/point_maze_v51_probe_residual_bank_v14_extension_20261022}"
FRESH_STATE_DATASET="${V15_FRESH_STATE_DATASET:-${ROOT}/data/point_maze_v52_probe_residual_bank_v15_fullscale_extension_20261022}"
CHECKPOINT="${ROOT}/data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"; DINOV2="${ROOT}/external/local_dinov2"
OUTPUT_DIR="${V15_OUTPUT_DIR:-${ROOT}/results/pointmaze_probe_residual_bank_v15_fullscale_20261022_cuda}"
METHOD="${V15_METHOD:-effect_adapter_rgb_online_probe_conditioned_residual_bank_verified_local}"
ROOT_MIN="${V15_ROOT_MIN:-18952}"; ROOT_MAX_EXCLUSIVE="${V15_ROOT_MAX_EXCLUSIVE:-19336}"; EXTENSION_SEED="${V15_EXTENSION_SEED:-202611180}"
FIT_ROOTS="1928,1929,1930,1931,1932,1933,1934,1935,1936,1937,1938,1939,1940,1941,1942,1943"
DESIGN_ROOTS="${V15_DESIGN_ROOTS:-$(seq -s, 18952 18983)}"; CAL_ROOTS="${V15_CAL_ROOTS:-$(seq -s, 18984 19015)}"; OUTCOME_ROOTS="${V15_OUTCOME_ROOTS:-$(seq -s, 19016 19079)}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "immutable V15 output already exists" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${CALIBRATION_DATASET}" "${SOURCE_STATE_DATASET}" "${CHECKPOINT}" "${DINOV2}"; do [[ -e "${required}" ]] || { echo "missing V15 input: ${required}" >&2; exit 3; }; done
for attempt in 1 2 3; do bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke && break; [[ "${attempt}" != 3 ]] || { echo "host CUDA preflight failed; refusing CPU fallback" >&2; exit 9; }; sleep 5; done
if [[ ! -e "${FRESH_STATE_DATASET}" ]]; then "${PYTHON}" -m "${EXTENDER}" --input-root "${SOURCE_STATE_DATASET}" --output "${FRESH_STATE_DATASET}" --episode-count $((ROOT_MAX_EXCLUSIVE - ROOT_MIN)) --seed "${EXTENSION_SEED}"; fi
MANIFEST="${FRESH_STATE_DATASET}/extension_manifest.json"
jq -e --argjson base "${ROOT_MIN}" --argjson extended "${ROOT_MAX_EXCLUSIVE}" --argjson count "$((ROOT_MAX_EXCLUSIVE - ROOT_MIN))" --argjson seed "${EXTENSION_SEED}" '.schema == "pointmaze_independent_state_action_extension.v1" and .base_root_count == $base and .extended_root_count == $extended and .new_episode_count == $count and .new_root_id_range == [$base, $extended - 1] and .seed == $seed' "${MANIFEST}" >/dev/null || { echo "V15 extension contract mismatch" >&2; exit 5; }
COMMON=(--dataset-root "${CALIBRATION_DATASET}" --test-state-root "${FRESH_STATE_DATASET}" --test-visual-source dataset_matched_renderer --checkpoint "${CHECKPOINT}" --dinov2-source "${DINOV2}" --calibration-episodes "${FIT_ROOTS}" --max-blocks 20 --directions 8 --magnitudes 0.25 0.50 0.75 1.00 --active-probe-directions 4 --active-probe-magnitude 1.00 --active-probe-trigger always --planner-map rgb_maze)
mkdir -p "${OUTPUT_DIR}/design" "${OUTPUT_DIR}/calibration"
echo "[V15] root-disjoint full-scale gate-design margins"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning "${COMMON[@]}" --test-episodes "${DESIGN_ROOTS}" --seed 202611181 --probe-conditioned-visual-margin 0.0 --probe-conditioned-min-advantage 0.0 --output "${OUTPUT_DIR}/design/margins.json" --methods "${METHOD}"
MARGIN_FREEZE="${OUTPUT_DIR}/frozen_design_margins.json"; "${PYTHON}" -m "${FREEZER}" --input "${OUTPUT_DIR}/design/margins.json" --output "${MARGIN_FREEZE}" --local-method "${METHOD}" --quantiles 0.25 0.50 0.75
echo "[V15] calibration baseline"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning "${COMMON[@]}" --test-episodes "${CAL_ROOTS}" --seed 202611182 --output "${OUTPUT_DIR}/calibration/visual.json" --methods visual_dynamics
SELECT_ARGS=(--baseline "${OUTPUT_DIR}/calibration/visual.json" --local-method "${METHOD}")
while IFS=$'\t' read -r label margin; do for advantage in 0.00 0.01 0.03; do
  name="${label}_a${advantage/./p}"; report="${OUTPUT_DIR}/calibration/${name}.json"; echo "[V15] calibration ${name} margin=${margin}"
  bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning "${COMMON[@]}" --test-episodes "${CAL_ROOTS}" --seed 202611182 --probe-conditioned-visual-margin "${margin}" --probe-conditioned-min-advantage "${advantage}" --output "${report}" --methods "${METHOD}"
  SELECT_ARGS+=(--candidate "${name}:${report}")
done; done < <(jq -r '.thresholds[] | "\(.label)\t\(.margin)"' "${MARGIN_FREEZE}")
SELECTION="${OUTPUT_DIR}/frozen_calibration_gate.json"; "${PYTHON}" -m "${SELECTOR}" "${SELECT_ARGS[@]}" --output "${SELECTION}"
AUTHORIZED="$(${PYTHON} -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["decision"]["deployment_authorized"])))' "${SELECTION}")"
if [[ "${AUTHORIZED}" != 1 ]]; then echo "[V15] no non-empty non-inferior full-scale local gate; outcome roots sealed."; exit 0; fi
LABEL="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"]["selected_label"])' "${SELECTION}")"
MARGIN="$(${PYTHON} -c 'import json,sys; x=json.load(open(sys.argv[1])); label=sys.argv[2]; print(next(t["margin"] for t in x["thresholds"] if label.startswith(t["label"])))' "${MARGIN_FREEZE}" "${LABEL}")"; ADVANTAGE="${LABEL##*_a}"; ADVANTAGE="${ADVANTAGE/p/.}"
echo "[V15] opening root-disjoint outcome under ${LABEL}"
bash "${WRAPPER}" "${PYTHON}" -m experiments.run_live_pointmaze_planning "${COMMON[@]}" --test-episodes "${OUTCOME_ROOTS}" --seed 202611183 --probe-conditioned-visual-margin "${MARGIN}" --probe-conditioned-min-advantage "${ADVANTAGE}" --output "${OUTPUT_DIR}/outcome/report.json" --methods visual_dynamics "${METHOD}"
