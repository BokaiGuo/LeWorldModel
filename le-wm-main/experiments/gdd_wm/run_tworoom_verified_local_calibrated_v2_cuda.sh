#!/usr/bin/env bash
# Fresh host-CUDA rerun of the corrected V1 contract.  V2 never reads the V1
# output directory: its 64 calibration and 128 outcome roots are disjoint from
# every V1 root, and it binds both seed panels into the selector/audit receipts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${ROOT}/.venv/bin/python"
WRAPPER="${ROOT}/experiments/run_with_torch_cudnn.sh"
RUNNER="experiments.run_live_tworoom_planning"
SELECTOR="experiments.gdd_wm.select_tworoom_verified_local_calibration"
AUDITOR="experiments.gdd_wm.audit_tworoom_verified_local_calibrated_v1"
PROTOCOL_ID="tworoom_verified_local_calibrated_v2_20261023"
OUTPUT="${ROOT}/results/gdd_wm_tworoom_verified_local_calibrated_v2_20261023_cuda"
ARTIFACT="${ROOT}/results/gdd_wm_goal_bilinear_teacher_20261020_cuda/capacity_readout.npz"
POLICY="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"

[[ ! -e "${OUTPUT}" ]] || { echo "immutable output exists: ${OUTPUT}" >&2; exit 2; }
for required in "${PYTHON}" "${WRAPPER}" "${ARTIFACT}" "${POLICY}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 3; }
done

# Sole CUDA admission check: no CPU fallback is permitted.
bash "${WRAPPER}" "${PYTHON}" -m experiments.gpu_preflight --device cuda --recurrent-smoke

BASE_COMMON=(--policy "${POLICY}" --device cuda --max-episode-steps 50
  --horizon 5 --receding-horizon 5 --action-block 5 --num-samples 64 --n-steps 8 --topk 8
  --candidate-bank --candidate-bank-variant vision_door16 --candidate-bank-iterations 7
  --hybrid-conflict-cos 0.0 --hybrid-model-min-norm 0.08
  --hybrid-use-effect-map --hybrid-use-effect-policy --hybrid-effect-map-intercept)
CALIBRATION_EPISODES=64
OUTCOME_EPISODES=128
CAL_SEED=20264224
OUTCOME_SEED=20264288
CAL_COMMON=("${BASE_COMMON[@]}" --episodes "${CALIBRATION_EPISODES}")
OUTCOME_COMMON=("${BASE_COMMON[@]}" --episodes "${OUTCOME_EPISODES}")
mkdir -p "${OUTPUT}/calibration"

run_pair() {
  local label="$1"; shift
  local teacher_extra=("$@")
  bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" "${CAL_COMMON[@]}" --seed "${CAL_SEED}" \
    --output "${OUTPUT}/calibration/${label}_baseline" --cost-metric euclidean "${teacher_extra[@]}"
  bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" "${CAL_COMMON[@]}" --seed "${CAL_SEED}" \
    --output "${OUTPUT}/calibration/${label}_teacher" --cost-metric capacity_safe_readout \
    --capacity-readout-artifact "${ARTIFACT}" --capacity-bilinear-mode full_predicted_dynamics "${teacher_extra[@]}"
}

run_pair verified_local
run_pair phase_local --hybrid-door-phase-reactive --hybrid-phase-transition-fallback
run_pair local_ranker --local-effect-ranker --local-effect-blend-weight 0.5

SELECTION="${OUTPUT}/frozen_calibration_selection.json"
"${PYTHON}" -m "${SELECTOR}" --output "${SELECTION}" --protocol-id "${PROTOCOL_ID}" \
  --expected-seed-start "${CAL_SEED}" --expected-root-count "${CALIBRATION_EPISODES}" \
  --candidate "verified_local:${OUTPUT}/calibration/verified_local_teacher/planning.json:${OUTPUT}/calibration/verified_local_baseline/planning.json" \
  --candidate "phase_local:${OUTPUT}/calibration/phase_local_teacher/planning.json:${OUTPUT}/calibration/phase_local_baseline/planning.json" \
  --candidate "local_ranker:${OUTPUT}/calibration/local_ranker_teacher/planning.json:${OUTPUT}/calibration/local_ranker_baseline/planning.json"

SELECTED="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_label"] or "")' "${SELECTION}")"
if [[ -z "${SELECTED}" ]]; then
  echo "calibration rejected all teacher contributions; outcome seeds remain sealed."
  "${PYTHON}" -m "${AUDITOR}" --output-dir "${OUTPUT}" --output "${OUTPUT}/outcome_audit.json" \
    --protocol-id "${PROTOCOL_ID}" --expected-seed-start "${OUTCOME_SEED}" --expected-root-count "${OUTCOME_EPISODES}"
  exit 0
fi

EXTRA=()
case "${SELECTED}" in
  verified_local) ;;
  phase_local) EXTRA=(--hybrid-door-phase-reactive --hybrid-phase-transition-fallback) ;;
  local_ranker) EXTRA=(--local-effect-ranker --local-effect-blend-weight 0.5) ;;
  *) echo "unknown frozen selection: ${SELECTED}" >&2; exit 4 ;;
esac
mkdir -p "${OUTPUT}/outcome"
bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" "${OUTCOME_COMMON[@]}" --seed "${OUTCOME_SEED}" \
  --output "${OUTPUT}/outcome/baseline" --cost-metric euclidean "${EXTRA[@]}"
bash "${WRAPPER}" "${PYTHON}" -m "${RUNNER}" "${OUTCOME_COMMON[@]}" --seed "${OUTCOME_SEED}" \
  --output "${OUTPUT}/outcome/teacher" --cost-metric capacity_safe_readout \
  --capacity-readout-artifact "${ARTIFACT}" --capacity-bilinear-mode full_predicted_dynamics "${EXTRA[@]}"
"${PYTHON}" -m "${AUDITOR}" --output-dir "${OUTPUT}" --output "${OUTPUT}/outcome_audit.json" \
  --protocol-id "${PROTOCOL_ID}" --expected-seed-start "${OUTCOME_SEED}" --expected-root-count "${OUTCOME_EPISODES}"
