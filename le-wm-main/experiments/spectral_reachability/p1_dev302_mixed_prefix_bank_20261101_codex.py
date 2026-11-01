"""RGB-only mixed prefix bank for the dev302 development root.

The bank is deterministic from the current and goal RGB plus sealed model
prefixes.  It contains zero, spectral, GN, three task-related contact programs,
and a fixed random control.  No simulator state or physical outcome is read.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import p1_dev302_rgb_contact_feedback_20261101_codex as rgb_source

GAIN = 2.0
PUSHER_RADIUS_PIXELS = 6.5
CLEARANCE_PIXELS = 10.0
RANDOM_SEED = 3021101


def require(value, message):
    if not value:
        raise ValueError(message)


def _features(current_rgb, goal_rgb):
    current = rgb_source.rgb._image(current_rgb)
    goal = rgb_source.rgb._image(goal_rgb)
    _, pusher = rgb_source.rgb._mask(current, rgb_source.rgb.BLUE, 8, single=False)
    object_mask, obj = rgb_source.rgb._mask(current, rgb_source.rgb.BLOCK, 8, single=False)
    _, goal_obj = rgb_source.rgb._mask(goal, rgb_source.rgb.BLOCK, 8, single=False)
    pusher_xy = np.asarray(pusher["centroid_xy"], np.float64)
    object_xy = np.asarray(obj["centroid_xy"], np.float64)
    goal_xy = np.asarray(goal_obj["centroid_xy"], np.float64)
    desired = goal_xy - object_xy
    norm = float(np.linalg.norm(desired))
    require(math.isfinite(norm) and norm > 1, "goal object displacement")
    desired /= norm
    py, px = np.nonzero(object_mask)
    pixels = np.stack((px, py), axis=1).astype(np.float64)
    deltas = pixels - pusher_xy
    nearest_index = int(np.argmin(np.linalg.norm(deltas, axis=1)))
    nearest_boundary = pixels[nearest_index]
    nearest_direction = nearest_boundary - pusher_xy
    nearest_direction /= np.linalg.norm(nearest_direction)
    support_index = int(np.argmin((pixels - object_xy) @ desired))
    support_boundary = pixels[support_index]
    support_target = support_boundary - desired * PUSHER_RADIUS_PIXELS
    nearest_target = nearest_boundary - nearest_direction * PUSHER_RADIUS_PIXELS
    return {
        "pusher_xy": pusher_xy,
        "object_xy": object_xy,
        "goal_object_xy": goal_xy,
        "desired_object_direction": desired,
        "nearest_direction": nearest_direction,
        "nearest_target": nearest_target,
        "support_target": support_target,
        "object_bounds": np.asarray(obj["bounds_xyxy"], np.float64),
    }


def _resample_polyline(start, waypoints, steps=5):
    points = np.vstack((np.asarray(start, np.float64), np.asarray(waypoints, np.float64)))
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    require(np.all(np.isfinite(segments)) and float(segments.sum()) > 0, "polyline")
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    targets = []
    for distance in np.linspace(segments.sum() / steps, segments.sum(), steps):
        index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(segments) - 1)
        fraction = (distance - cumulative[index]) / segments[index]
        targets.append(points[index] + fraction * (points[index + 1] - points[index]))
    targets = np.asarray(targets)
    deltas = np.diff(np.vstack((points[0], targets)), axis=0)
    raw = deltas * (512.0 / 224.0) / 100.0 * GAIN
    return np.clip(raw, -1, 1).astype(np.float32)


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    features = _features(current_rgb, goal_rgb)
    spectral = np.asarray(spectral_prefix, np.float32)
    gn = np.asarray(gn_prefix, np.float32)
    require(
        spectral.shape == gn.shape == (5, 2)
        and np.isfinite(spectral).all() and np.isfinite(gn).all()
        and np.max(np.abs(spectral)) <= 1 and np.max(np.abs(gn)) <= 1,
        "sealed prefixes",
    )
    x0, y0, x1, y1 = features["object_bounds"]
    support = features["support_target"]
    top_right = np.asarray([x1 + CLEARANCE_PIXELS, y0 - CLEARANCE_PIXELS])
    bottom_left = np.asarray([x0 - CLEARANCE_PIXELS, y1 + CLEARANCE_PIXELS])
    nearest = _resample_polyline(features["pusher_xy"], [features["nearest_target"]])
    support_direct = _resample_polyline(features["pusher_xy"], [support])
    support_top = _resample_polyline(features["pusher_xy"], [top_right, support])
    support_bottom = _resample_polyline(features["pusher_xy"], [bottom_left, support])
    rng = np.random.default_rng(RANDOM_SEED)
    random = rng.uniform(-0.75, 0.75, size=(5, 2)).astype(np.float32)
    candidates = [
        ("zero", np.zeros((5, 2), np.float32), False),
        ("spectral", spectral, None),
        ("gn", gn, None),
        ("rgb_nearest_contact", nearest, True),
        ("rgb_goal_support_direct", support_direct, True),
        ("rgb_goal_support_top_route", support_top, True),
        ("rgb_goal_support_bottom_route", support_bottom, True),
        ("random_control", random, None),
    ]
    direction = features["nearest_direction"].astype(np.float32)
    rows = []
    for index, (label, actions, constructed) in enumerate(candidates):
        projection = float(np.sum(actions[:2] @ direction))
        feasible = bool(constructed) if constructed is not None else projection >= 0.05
        rows.append({
            "index": index,
            "label": label,
            "actions": actions.tolist(),
            "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
            "nearest_contact_projection_first2": projection,
            "contact_feasible_prior": feasible,
            "constructed_from_rgb_contact_geometry": bool(constructed),
        })
    return {
        "schema": "p1_dev302_mixed_prefix_bank_20261101_codex.v1",
        "candidates": rows,
        "geometry": {
            key: value.tolist()
            for key, value in features.items()
            if key != "object_bounds"
        } | {"object_bounds_xyxy": features["object_bounds"].tolist()},
        "candidate_order_fixed": True,
        "simulator_state_used": False,
        "physical_outcome_used": False,
    }
