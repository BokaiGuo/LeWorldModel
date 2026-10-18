#!/usr/bin/env bash
set -euo pipefail

# Host-side entry point for the frozen Cube repair protocol.
# The caller must run this from a host shell with /dev/nvidia* visible.
# No CPU fallback is permitted.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="${OUTPUT:-$ROOT/results/cube_gpu_repair_panel_20261018}"

# This launcher is itself host-only.  Keep the explicit preflight below for
# the panel artifact, while the guard provides the durable device/driver/EGL
# boundary for direct host invocation.
if [[ -z "${LEWM_HOST_GPU_GUARD_ACTIVE:-}" ]]; then
  export LEWM_HOST_GPU_GUARD_ACTIVE=1
  exec "$ROOT/scripts/host_gpu_exec.sh" "$0" "$@"
fi

# Keep the MuJoCo renderer deterministic in a non-interactive host shell.
# This is an execution prerequisite, not a scientific intervention.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1

RESUME_ARGS=()
if [[ -f "$OUTPUT/ceiling.json" ]] && grep -q '"schema": "cube_planner_ceiling.v2.partial"' "$OUTPUT/ceiling.json"; then
  RESUME_ARGS=(--resume)
  echo "[resume] continuing partial ceiling checkpoint: $OUTPUT/ceiling.json"
fi
PANEL_RESUME_ARGS=()
if compgen -G "$OUTPUT/panel/*.json" > /dev/null && grep -R -q '"schema": "live_cube_rgb_macro_ranker.v1"' "$OUTPUT/panel"; then
  PANEL_RESUME_ARGS=(--resume)
  echo "[resume] continuing completed RGB arms under: $OUTPUT/panel"
fi

"$PYTHON" -m experiments.gpu_preflight \
  --device cuda \
  --render-smoke \
  --output "$OUTPUT/cuda_preflight.json"

"$PYTHON" -m experiments.audit_cube_planner_ceiling \
  --output "$OUTPUT/ceiling.json" \
  --seed 20261900 \
  --test-roots 8 \
  --macro-length 40 \
  --expanded-macro-length 40 \
  --goal-tolerance-m 0.05 \
  --descent-steps 15 \
  --approach-steps 8 \
  --approach-calibration-roots 40 \
  --probe-steps 4 \
  --macro-cycles 8 \
  "${RESUME_ARGS[@]}"

"$PYTHON" -m experiments.run_live_cube_rgb_macro_panel \
  --device cuda \
  --output "$OUTPUT/panel" \
  --seed 20261900 \
  --calibration-roots 40 \
  --test-roots 8 \
  --macro-cycles 8 \
  --action-block 1 \
  --approach-steps 8 \
  --descent-steps 15 \
  --probe-steps 4 \
  --calibration-target rgb_goal \
  --contact-aware-features \
  --arms fullscale_reclose lowgain_backtrack lowgain_mode_reclose factorized_lowgain_backtrack multiphase_lowgain_backtrack \
  "${PANEL_RESUME_ARGS[@]}"

"$PYTHON" -m experiments.summarize_cube_failure_cause \
  --ceiling "$OUTPUT/ceiling.json" \
  --panel "$OUTPUT/panel/panel.json" \
  --output "$OUTPUT/failure_cause.json"

echo "Cube GPU repair panel complete: $OUTPUT"
