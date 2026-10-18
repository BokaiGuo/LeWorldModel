#!/usr/bin/env bash
set -euo pipefail

# Fresh-root paired intervention for the remaining Cube bottleneck:
# root-local RGB action-effect calibration versus the chart baseline.
# This is a host-only formal experiment and never falls back to CPU.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/cube_action_effect_panel_20261018}"

if [[ -z "${LEWM_HOST_GPU_GUARD_ACTIVE:-}" ]]; then
  export LEWM_HOST_GPU_GUARD_ACTIVE=1
  exec "$ROOT/scripts/host_gpu_exec.sh" "$0" "$@"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT"

"$PYTHON" -m experiments.gpu_preflight \
  --device cuda --render-smoke \
  --output "$OUTPUT/cuda_preflight.json"

COMMON=(
  --seed 20261901 --tasks 1 2
  --calibration-roots 40 --test-roots 8
  --descent-steps 15 --approach-steps 32 --approach-controller pixel
  --local-pixel-approach --fit-device cuda
  --close-steps 5 --lift-before-probes --lift-steps 15
  --probe-magnitude 0.5
  --transport-steps 50 --transport-magnitude 0.5
  --lower-steps 15 --release-steps 8
  --goal-threshold 0.05
)

BASELINE="${BASELINE_INPUT:-$OUTPUT/chart.json}"
if [[ -z "${BASELINE_INPUT:-}" && ! -s "$BASELINE" ]]; then
  "$PYTHON" -m experiments.run_live_cube_rgb_pick_place \
    "${COMMON[@]}" --transport-controller chart --output "$BASELINE"
fi

for controller in effect hybrid; do
  target="$OUTPUT/${controller}.json"
  if [[ ! -s "$target" ]]; then
    "$PYTHON" -m experiments.run_live_cube_rgb_pick_place \
      "${COMMON[@]}" \
      --initial-conditions "$BASELINE" \
      --transport-controller "$controller" \
      --output "$target"
  fi
done

"$PYTHON" -m experiments.summarize_cube_action_effect_panel \
  --input "$BASELINE" \
  --input "$OUTPUT/effect.json" \
  --input "$OUTPUT/hybrid.json" \
  --output "$OUTPUT/summary.json"

echo "Cube action-effect panel complete: $OUTPUT"
