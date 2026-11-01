"""V4 RGB-only contact guard with finite-horizon route completion margin.

V3 exposed a development-only false positive: the five-step top route ended at
the ideal raster support point but remained just outside native contact.  V4
keeps direct and non-RGB candidates unchanged, extends only routed candidates
three pixels through the support boundary, and applies a stricter RGB path
guard.  No simulator state or physical outcome is read by ``build``.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import p1_dev302_mixed_prefix_bank_v3_20261101_codex as v3
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry

ROUTE_TERMINAL_OVERTRAVEL_PIXELS = 3.0
CONTACT_CENTER_TO_OBJECT_PIXELS = 5.5


def _replace_routed_actions(result, current_rgb, goal_rgb):
    features = geometry._features(current_rgb, goal_rgb)
    x0, y0, x1, y1 = features["object_bounds"]
    routed_support = (
        features["support_target"]
        + features["desired_object_direction"] * ROUTE_TERMINAL_OVERTRAVEL_PIXELS
    )
    waypoints = {
        "rgb_goal_support_top_route": [
            np.asarray([x1 + geometry.CLEARANCE_PIXELS, y0 - geometry.CLEARANCE_PIXELS]),
            routed_support,
        ],
        "rgb_goal_support_bottom_route": [
            np.asarray([x0 - geometry.CLEARANCE_PIXELS, y1 + geometry.CLEARANCE_PIXELS]),
            routed_support,
        ],
    }
    for row in result["candidates"]:
        points = waypoints.get(row["label"])
        if points is None:
            continue
        actions = geometry._resample_polyline(features["pusher_xy"], points)
        row["actions"] = actions.tolist()
        row["actions_sha256"] = hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()
        row["route_terminal_overtravel_pixels"] = ROUTE_TERMINAL_OVERTRAVEL_PIXELS
    return features


def _recompute_guard(result, current_rgb):
    image = geometry.rgb_source.rgb._image(current_rgb)
    object_mask, _ = geometry.rgb_source.rgb._mask(
        image, geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    for row in result["candidates"]:
        actions = np.asarray(row["actions"], np.float64)
        centers = start + np.cumsum(actions, axis=0) * v3.PIXELS_PER_CUMULATIVE_RAW
        gaps = np.min(
            np.linalg.norm(centers[:, None, :] - object_pixels[None, :, :], axis=-1),
            axis=1,
        )
        minimum = float(gaps.min())
        row["rgb_calibrated_predicted_pusher_centers_xy"] = centers.tolist()
        row["rgb_calibrated_object_boundary_gap_by_step"] = gaps.tolist()
        row["rgb_calibrated_minimum_object_boundary_gap"] = minimum
        row["contact_feasible_prior"] = minimum <= CONTACT_CENTER_TO_OBJECT_PIXELS


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v3.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    _replace_routed_actions(result, current_rgb, goal_rgb)
    _recompute_guard(result, current_rgb)
    result["schema"] = "p1_dev302_mixed_prefix_bank_v4_20261101_codex.v1"
    result["contact_guard"] = {
        "kind": "rgb_calibrated_action_path_with_routed_terminal_overtravel",
        "pixels_per_cumulative_raw": v3.PIXELS_PER_CUMULATIVE_RAW,
        "contact_center_to_object_pixels": CONTACT_CENTER_TO_OBJECT_PIXELS,
        "route_terminal_overtravel_pixels": ROUTE_TERMINAL_OVERTRAVEL_PIXELS,
        "direct_candidate_changed": False,
        "simulator_state_used": False,
        "calibration": (
            "development RGB trajectories only; routed endpoint margin fixed "
            "after the sealed V3 false-positive diagnosis"
        ),
        "prospective_status": "requires new development holdout",
    }
    return result
