"""Five-step actual-feedback boundary for an AP-compatible development runner.

No model, optimizer, environment construction, reset, or scientific gate.
The caller owns provenance, lock, durable event sink, and neural planning.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


def _rgb(value):
    x = np.asarray(value)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3 or min(x.shape) < 1:
        raise ValueError("render must return nonempty uint8 HWC RGB")
    return np.ascontiguousarray(x).copy()


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def execute_block(env, actions, *, expected_root, expected_goal, goal, emit):
    """Execute one block and return a feedback handoff or explicit partial.

    ``emit`` must durably persist each event or raise.  An intent is emitted
    before each irreversible callback; exceptions stop the block.  It is the
    caller's responsibility to preserve sink failures and use the proper lock.
    The incoming root is an actual prior observation, never freshly inferred.
    """
    a = np.asarray(actions)
    if a.dtype != np.float32 or a.ndim != 2 or a.shape[0] != 5 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError("actions must be finite float32 [5,action_dim]")
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    if (low.shape != (a.shape[1],) or high.shape != low.shape or
            not np.isfinite(low).all() or not np.isfinite(high).all() or
            np.any(low >= high) or np.any(a < low) or np.any(a > high)):
        raise ValueError("actions or bounds invalid")
    root, goal_image = _rgb(expected_root), _rgb(goal)
    if _hash(goal_image) != expected_goal:
        raise ValueError("goal hash mismatch")
    receipt = {"status": "running", "root_rgb_sha256": _hash(root),
               "goal_rgb_sha256": expected_goal, "planned_action_sha256": _hash(a),
               "attempted": 0, "returned": 0, "steps": [], "failure": None,
               "successor_rgb_sha256": None}
    emit({"kind": "block_open", "root_rgb_sha256": receipt["root_rgb_sha256"],
          "goal_rgb_sha256": expected_goal, "planned_action_sha256": receipt["planned_action_sha256"]})
    for index, action in enumerate(a):
        intent = {"kind": "step_intent", "step": index + 1, "action_float32_hex": action.tobytes().hex()}
        emit(intent)
        receipt["attempted"] += 1
        try:
            result = env.step(action.copy())
            if not isinstance(result, tuple) or len(result) != 5:
                raise ValueError("step must return Gymnasium 5-tuple")
            _, reward, terminated, truncated, _ = result
            if (not isinstance(terminated, (bool, np.bool_)) or
                    not isinstance(truncated, (bool, np.bool_)) or
                    isinstance(reward, (bool, np.bool_)) or not math.isfinite(float(reward))):
                raise ValueError("invalid step return labels")
            record = {"step": index + 1, "action_float32_hex": action.tobytes().hex(),
                      "reward": float(reward), "terminated": bool(terminated),
                      "truncated": bool(truncated)}
        except Exception as exc:
            receipt.update(status="step_failed", failure={"step": index + 1,
                           "type": type(exc).__name__, "message": str(exc)})
            emit({"kind": "step_failed", **receipt["failure"]})
            return receipt, None
        receipt["returned"] += 1
        receipt["steps"].append(record)
        emit({"kind": "step_returned", **record})
        if record["terminated"] or record["truncated"]:
            receipt["status"] = "native_stopped"
            emit({"kind": "block_stopped", "step": index + 1,
                  "terminated": record["terminated"], "truncated": record["truncated"]})
            return receipt, None
    emit({"kind": "render_intent", "after_returned_steps": receipt["returned"]})
    try:
        successor = _rgb(env.render())
        if successor.shape != root.shape:
            raise ValueError("successor/root RGB shapes differ")
    except Exception as exc:
        receipt.update(status="render_failed", failure={"type": type(exc).__name__, "message": str(exc)})
        emit({"kind": "render_failed", **receipt["failure"]})
        return receipt, None
    receipt["successor_rgb_sha256"] = _hash(successor)
    receipt["status"] = "complete"
    emit({"kind": "feedback_ready", "root_rgb_sha256": receipt["root_rgb_sha256"],
          "successor_rgb_sha256": receipt["successor_rgb_sha256"],
          "executed_action_sha256": receipt["planned_action_sha256"]})
    return receipt, {"root_frame": successor, "goal": goal_image,
                     "executed_actions": a.copy(), "root_rgb_sha256": receipt["successor_rgb_sha256"],
                     "goal_rgb_sha256": expected_goal}
