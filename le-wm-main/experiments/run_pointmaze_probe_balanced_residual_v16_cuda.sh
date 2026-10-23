#!/usr/bin/env bash
# V16 changes only information geometry: map-safe balanced rank-2 probes over
# the full declared planner bank.  V15's calibration/outcome roots remain
# immutable; this wrapper supplies a root-disjoint panel to the hardened
# three-way calibration protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env \
  V15_SOURCE_STATE_DATASET="${ROOT}/data/point_maze_v52_probe_residual_bank_v15_fullscale_extension_20261022" \
  V15_FRESH_STATE_DATASET="${ROOT}/data/point_maze_v53_probe_balanced_residual_v16_extension_20261022" \
  V15_OUTPUT_DIR="${ROOT}/results/pointmaze_probe_balanced_residual_v16_20261022_cuda" \
  V15_METHOD="effect_adapter_rgb_online_probe_conditioned_balanced_residual_bank_verified_local" \
  V15_ROOT_MIN=19336 V15_ROOT_MAX_EXCLUSIVE=19720 V15_EXTENSION_SEED=202611190 \
  V15_DESIGN_ROOTS="$(seq -s, 19336 19367)" \
  V15_CAL_ROOTS="$(seq -s, 19368 19399)" \
  V15_OUTCOME_ROOTS="$(seq -s, 19400 19463)" \
  bash "${ROOT}/experiments/run_pointmaze_probe_residual_bank_v15_cuda.sh"
