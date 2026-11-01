"""Prospective RGB feedback candidate, not a physical or CUDA launcher.

Called only after actual phase-one execution and fresh RGB authentication. It
accepts no simulator state, rewards, future RGB, or historical selected action.
This fixed heuristic is post-outcome exploration, not a trained RGB surrogate.
"""
import hashlib
import math

import numpy as np

from . import codex_p1_rgb_task_score_20261030_v2 as rgb

POLICY = {
    "schema": "codex_p1_rgb_feedback_candidate_20261030_v1",
    "rgb_moment": "fresh_after_first_executed_5_suffix_actions",
    "object_shape_chamfer_max_pixels": 2.0,
    "pusher_distance_min_pixels": 8.0,
    "correction_raw_magnitude": 0.2,
    "correction_schedule": "one_relative_action_then_four_zero",
    "otherwise": "retain_current_stage_model_first_packet",
    "rgb_rejection": "no_second_stage_action",
    "native_stop": "no_later_planning_or_action",
}


def decide(current_rgb, goal_rgb, model_first_packet):
    """Select one raw 5x2 packet from observed RGB and current model packet.

    Object and pusher measurements stay separate. Any segmentation failure is
    returned as a rejection, not silently replaced by a baseline action.
    """
    if (not isinstance(model_first_packet, np.ndarray) or model_first_packet.dtype != np.dtype("<f4")
            or model_first_packet.shape != (5, 2) or not np.isfinite(model_first_packet).all()
            or not (np.abs(model_first_packet) <= 1).all()):
        raise ValueError("model packet ABI")
    try:
        features = rgb.score(current_rgb, goal_rgb)
    except ValueError as exc:
        return {"decision": "rgb_rejected", "reason": str(exc), "packet": None, "features": None}
    packet = model_first_packet.copy()
    decision = "model_packet"
    if (features["object_shape_chamfer_pixels"] <= POLICY["object_shape_chamfer_max_pixels"] and
            features["pusher_centroid_distance_pixels"] > POLICY["pusher_distance_min_pixels"]):
        current = np.asarray(features["current_pusher"]["centroid_xy"], np.float64)
        goal = np.asarray(features["goal_pusher"]["centroid_xy"], np.float64)
        delta = goal-current
        norm = float(np.linalg.norm(delta))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("pusher direction")
        packet = np.zeros((5, 2), dtype="<f4")
        packet[0] = (delta/norm * POLICY["correction_raw_magnitude"]).astype("<f4")
        decision = "rgb_pusher_correction"
    return {"decision": decision, "reason": None, "packet": packet,
            "packet_sha256": hashlib.sha256(packet.tobytes()).hexdigest(), "features": features}
