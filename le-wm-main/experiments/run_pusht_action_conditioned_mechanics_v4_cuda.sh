#!/usr/bin/env bash
# Fresh host-CUDA confirmation of the declared RGB contact/action mechanics readout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
OUTPUT="${ROOT}/results/pusht_action_conditioned_mechanics_v4_20261023_cuda"
RUNNER="experiments.run_pusht_action_conditioned_signed_torque_confirmation"
AUDITOR="experiments.audit_pusht_action_conditioned_signed_torque_confirmation"

[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to reuse immutable protocol output: ${OUTPUT}" >&2; exit 2; }
mkdir -p "${OUTPUT}"
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight --device cuda --recurrent-smoke
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m "${RUNNER}" \
  --output "${OUTPUT}/action_conditioned_signed_torque_confirmation.json" --device cuda \
  --train-seed-band 1704240000 --calibration-seed-band 1704250000 --test-seed-band 1704260000 \
  --train-per-class 128 --calibration-per-class 64 --test-per-class 128 \
  --confirmation-representation geometry_plus_pair_mechanics
.venv/bin/python -m "${AUDITOR}" \
  --report "${OUTPUT}/action_conditioned_signed_torque_confirmation.json" \
  --output "${OUTPUT}/report_consistency_audit.json"
