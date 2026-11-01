"""Fixed two-pass GPU rendering; no cached, rounded or outcome-chosen pixels.

The first actual draw is always discarded. The second actual draw is returned.
Every draw must preserve the complete integration/model/scene state. This is an
explicit observation runtime policy, not a model repair or P1 admission.
"""
from __future__ import annotations

import numpy as np

from experiments.spectral_reachability import reacher_d2p_context as execution

POLICY = "fixed-warm-render-v1"
REQUIRED = ("qpos", "qvel", "physics_state", "physics_integration_state", "integration_state_spec",
            "model_binary_sha256", "target_qpos", "target_marker_pos")


class FixedWarmRenderer:
    def __init__(self, env, *, capture=execution.capture_execution_snapshot, touch_gl=None):
        self.env, self.raw, self.capture = env, env.unwrapped, capture
        self.original = self.raw.render
        if touch_gl is None:
            from OpenGL import GL
            def touch_gl():
                with self.raw.env.physics.contexts.gl.make_current() as context:
                    return bool(context.call(GL.glIsEnabled, GL.GL_DITHER))
        self.touch_gl = touch_gl
        self.stats = {"policy": POLICY, "requests": 0, "raw_render_passes": 0,
            "exact_state_checks": 0, "warmup_retained_differences": 0,
            "max_warmup_retained_pixel_error": 0, "dither_state_changes": 0}
        self.wrapped = self._render
        self.raw.render = self.wrapped

    def _state(self):
        state = self.capture(self.env)
        if not isinstance(state, dict) or any(k not in state for k in REQUIRED):
            raise ValueError("complete MuJoCo integration/model/scene snapshot required")
        return state

    def _render(self, *args, **kwargs):
        before_gl = self.touch_gl()
        before = self._state()
        warm = np.asarray(self.original(*args, **kwargs)).copy()
        self.stats["raw_render_passes"] += 1
        if self._state() != before: raise RuntimeError("warm draw changed complete physical/scene state")
        self.stats["exact_state_checks"] += 1
        retained = np.asarray(self.original(*args, **kwargs)).copy()
        self.stats["raw_render_passes"] += 1
        if self._state() != before: raise RuntimeError("retained draw changed complete physical/scene state")
        self.stats["exact_state_checks"] += 1
        after_gl = self.touch_gl()
        if (warm.dtype != np.uint8 or retained.dtype != np.uint8 or warm.ndim != 3
                or warm.shape != retained.shape or warm.shape[-1] != 3):
            raise ValueError("unaltered uint8 HWC RGB draws required")
        error = int(np.max(np.abs(warm.astype(np.int16)-retained.astype(np.int16))))
        self.stats["requests"] += 1
        self.stats["warmup_retained_differences"] += int(error > 0)
        self.stats["max_warmup_retained_pixel_error"] = max(self.stats["max_warmup_retained_pixel_error"], error)
        self.stats["dither_state_changes"] += int(before_gl != after_gl)
        return retained

    def restore(self):
        if self.raw.render is not self.wrapped: raise RuntimeError("render policy was replaced unexpectedly")
        self.raw.render = self.original


def valid_receipts(receipts, seeds):
    seeds = list(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds): return False
    if not isinstance(receipts, list) or any(not isinstance(r, dict) for r in receipts): return False
    if [r.get("seed") for r in receipts] != list(seeds): return False
    for r in receipts:
        s = r.get("stats", {})
        keys = ("requests", "raw_render_passes", "exact_state_checks", "warmup_retained_differences",
                "max_warmup_retained_pixel_error", "dither_state_changes")
        if s.get("policy") != POLICY or any(type(s.get(k)) is not int or s[k] < 0 for k in keys): return False
        if (s["requests"] < 1 or s["raw_render_passes"] != 2*s["requests"]
                or s["exact_state_checks"] != 2*s["requests"]
                or s["warmup_retained_differences"] > s["requests"]
                or s["max_warmup_retained_pixel_error"] > 255 or s["dither_state_changes"] != 0): return False
    return True
