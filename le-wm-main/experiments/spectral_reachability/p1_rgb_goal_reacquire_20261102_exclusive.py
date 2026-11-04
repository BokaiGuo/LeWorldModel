"""Conservative RGB goal acquisition barrier. This module never dispatches actions."""
import hashlib
from pathlib import Path

import numpy as np

from . import p1_rgb_visibility_20261101_main as guard

GUARD_SHA256 = "a88c6587961ec491d7408814d0ab87b621e605246c74ebbea0785a2795472381"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _guard_intact():
    return _sha(Path(guard.__file__).read_bytes()) == GUARD_SHA256


def _blocked(reason, **details):
    return {"status": "blocked", "reason": reason, "action": None,
            "selector_called": False, "details": details}


def choose(goal_captures, current, *, now_ns, max_age_ns, selector):
    """Call selector(goal RGB, current RGB) only after all acquisition checks pass.

    Metadata must be supplied by a trusted acquisition boundary; it is not
    authenticated by this function. No data from simulator state is accepted.
    """
    selector_called = False
    try:
        if not _guard_intact():
            return _blocked("guard_source_mismatch")
        if not isinstance(goal_captures, (list, tuple)) or not goal_captures:
            return _blocked("goal_capture_missing")
        first = goal_captures[0]
        first_screen = guard.inspect(first["pixels"])
        if not first_screen["eligible"]:
            return _blocked("goal_visibility_failed", goal_screen=first_screen)
        if len(goal_captures) != 2:
            return _blocked("two_goal_acquisitions_required")
        second = goal_captures[1]
        second_screen = guard.inspect(second["pixels"])
        current_screen = guard.inspect(current["pixels"])
        if not second_screen["eligible"]:
            return _blocked("reacquired_goal_visibility_failed", goal_screen=second_screen)
        if not current_screen["eligible"]:
            return _blocked("current_visibility_failed", current_screen=current_screen)
        if (type(now_ns) is not int or type(max_age_ns) is not int or
                now_ns < 0 or max_age_ns <= 0):
            return _blocked("invalid_freshness_policy")
        captures = (first, second, current)
        ids = [x["capture_id"] for x in captures]
        seq = [x["sequence"] for x in captures]
        times = [x["captured_ns"] for x in captures]
        if (any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 3 or
                any(type(x) is not int or x < 0 for x in seq + times) or
                not seq[0] < seq[1] < seq[2] or not times[0] < times[1] <= times[2] or
                any(t > now_ns or now_ns - t > max_age_ns for t in times)):
            return _blocked("capture_identity_order_or_freshness_failed")
        for key in ("session_id", "camera_id", "target_id"):
            values = [x[key] for x in captures]
            if (any(not isinstance(v, str) or not v for v in values) or
                    len(set(values)) != 1):
                return _blocked("capture_context_mismatch", field=key)
        m1, m2 = guard.masks(first["pixels"]), guard.masks(second["pixels"])
        if any(not np.array_equal(m1[name], m2[name]) for name in ("object", "pusher")):
            return _blocked("goal_masks_disagree")
        # Copies prevent a selector from changing source frames.
        selector_called = True
        selected = selector(first["pixels"].copy(), current["pixels"].copy())
        action = np.asarray(selected, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            return {"status": "blocked", "reason": "invalid_selector_output", "action": None,
                    "selector_called": True, "details": {}}
        return {"status": "ready", "reason": None, "action": action.tolist(),
                "selector_called": True, "details": {
                    "goal_pixel_sha256": _sha(first["pixels"].tobytes()),
                    "capture_ids": ids, "goal_screen": first_screen,
                    "current_screen": current_screen}}
    except Exception as exc:
        # Includes malformed inputs and selector exceptions. Never substitute zero action:
        # a zero command is still a physical action.
        return {"status": "blocked", "reason": "input_or_selector_error", "action": None,
                "selector_called": selector_called, "details": {"exception_type": type(exc).__name__}}
