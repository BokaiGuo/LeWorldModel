"""Development-only RGB edge and candidate approach diagnostic.

This is not a selector, action generator, physical safety certificate, or
modification of any sealed protocol. Predictions assume a stationary object and
the frozen non-contact motion approximation; contact can invalidate that model.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import codex_p1_rgb_task_score_20261030_v2 as rgb
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry
from . import p1_dev302_mixed_prefix_bank_v5_20261101_codex as dynamics
from . import p1_dev302_mixed_prefix_bank_v6_20261101_codex as support


NAME = "p1_rgb_edge_approach_diagnostic_20261102_exclusive"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inside(points, width, height, radius):
    points = np.asarray(points, dtype=np.float64)
    return ((points[..., 0] >= radius) & (points[..., 0] <= width - 1 - radius)
            & (points[..., 1] >= radius) & (points[..., 1] <= height - 1 - radius))


def _candidate(row, start, pixels, desired, width, height):
    require(type(row) is dict and set(row) == {"label", "actions"}, "candidate fields")
    require(isinstance(row["label"], str) and bool(row["label"]), "candidate label")
    actions = np.asarray(row["actions"], dtype=np.float64)
    require(actions.shape == (5, 2) and np.isfinite(actions).all()
            and (np.abs(actions) <= 1).all(), "bounded finite five-step actions")
    actions32 = actions.astype("<f4")
    # Use exactly the float32 command sequence that would be dispatched.
    centers = dynamics._centers(start, actions32)
    require(np.isfinite(centers).all(), "finite predicted centers")
    points = np.vstack((start, centers))
    safe = _inside(points, width, height, geometry.PUSHER_RADIUS_PIXELS)
    gaps = np.min(np.linalg.norm(points[:, None, :] - pixels[None, :, :], axis=-1), axis=1)
    premature_contact = bool(np.any(gaps[:-1] <= dynamics.CONTACT_CENTER_TO_OBJECT_PIXELS))
    delta = pixels - centers[-1]
    distances = np.linalg.norm(delta, axis=1)
    nearest = int(np.argmin(distances))
    gap = float(distances[nearest])
    alignment = None if gap == 0 else float(delta[nearest] @ desired / gap)
    contact = bool(gap <= dynamics.CONTACT_CENTER_TO_OBJECT_PIXELS)
    aligned = alignment is not None and alignment >= support.MIN_SUPPORT_ALIGNMENT
    return {
        "label": row["label"],
        "actions_sha256": hashlib.sha256(actions32.tobytes()).hexdigest(),
        "predicted_centers_xy": centers.tolist(),
        "predicted_center_and_disk_in_frame_by_step": safe.tolist(),
        "predicted_edge_safe": bool(safe.all()),
        "final_nearest_object_pixel_xy": pixels[nearest].tolist(),
        "final_object_gap_pixels": gap,
        "final_support_alignment": alignment,
        "frozen_contact_distance_pass": contact,
        "frozen_support_alignment_pass": bool(aligned),
        "predicted_contact_before_final_step": premature_contact,
        "approach_feasible_approximation": bool(
            safe.all() and not premature_contact and contact and aligned),
    }


def diagnose(current_rgb, goal_rgb, candidates):
    """Assess provided actions from RGB only; never select or alter candidates.

    Inputs must be renderer-native RGB224 and a list of dicts containing only
    ``label`` and five-by-two ``actions``. Clipped/absent masks and undefined
    goal displacement fail closed. Results are not a holdout confirmation.
    """
    current = rgb._image(current_rgb)
    goal = rgb._image(goal_rgb)
    _, pusher = rgb._mask(current, rgb.BLUE, 8, single=False)
    object_mask, obj = rgb._mask(current, rgb.BLOCK, 8, single=False)
    _, goal_obj = rgb._mask(goal, rgb.BLOCK, 8, single=False)
    start = np.asarray(pusher["centroid_xy"], dtype=np.float64)
    object_center = np.asarray(obj["centroid_xy"], dtype=np.float64)
    goal_center = np.asarray(goal_obj["centroid_xy"], dtype=np.float64)
    displacement = goal_center - object_center
    norm = float(np.linalg.norm(displacement))
    require(math.isfinite(norm) and norm > 1.0, "undefined goal object direction")
    desired = displacement / norm
    py, px = np.nonzero(object_mask)
    pixels = np.column_stack((px, py)).astype(np.float64)
    support_pixel = pixels[int(np.argmin((pixels - object_center) @ desired))]
    target = support_pixel - desired * geometry.PUSHER_RADIUS_PIXELS
    height, width = current.shape[:2]
    require(isinstance(candidates, (list, tuple)) and len(candidates) > 0,
            "nonempty candidate list")
    rows = [_candidate(row, start, pixels, desired, width, height) for row in candidates]
    require(len({row["label"] for row in rows}) == len(rows), "unique candidate labels")
    return {
        "schema": NAME + ".v1",
        "classification": "development_diagnostic_not_original_holdout_confirmation",
        "P1_admission": False,
        "independent_confirmation": False,
        "simulator_state_used": False,
        "physical_outcomes_read": False,
        "candidate_actions_changed": False,
        "thresholds_source": "frozen_v5_v6_and_rgb_geometry",
        "pusher_radius_pixels": geometry.PUSHER_RADIUS_PIXELS,
        "contact_distance_pixels": dynamics.CONTACT_CENTER_TO_OBJECT_PIXELS,
        "minimum_support_alignment": support.MIN_SUPPORT_ALIGNMENT,
        "object_bounds_xyxy": obj["bounds_xyxy"],
        "support_boundary_pixel_xy": support_pixel.tolist(),
        "support_target_xy": target.tolist(),
        "support_target_center_in_frame": bool(_inside(target, width, height, 0.0)),
        "support_target_disk_in_frame": bool(
            _inside(target, width, height, geometry.PUSHER_RADIUS_PIXELS)),
        "candidates": rows,
        "limitations": [
            "exact-color segmentation and image bounds are not physical workspace bounds",
            "frozen motion model assumes no contact and can be wrong",
            "stationary object and step endpoints only; no occlusion or collision guarantee",
            "not an action selector or authorization to alter a sealed contract",
        ],
    }
