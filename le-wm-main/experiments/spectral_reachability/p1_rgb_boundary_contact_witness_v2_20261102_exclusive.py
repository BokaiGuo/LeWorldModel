"""New RGB-only boundary-support approach protocol; no simulator or outcomes.

The caller supplies a separately frozen and SHA-256-pinned development motion
configuration. A returned witness is only a model-predicted, pixel-geometric
approach, not evidence of physical contact, safety, or task success.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

NAME = "p1_rgb_boundary_contact_witness_v2_20261102_exclusive"
SCHEMA = NAME + ".v1"
CONFIG_SCHEMA = "p1_rgb_boundary_contact_witness_20261102_exclusive.config.v1"
OBJECT = (119, 136, 153)
PUSHER = (65, 105, 225)
ANGLES_DEG = (0, -30, 30, -45, 45, -60, 60)
# Frozen limits of p1_rgb_chunk_approach_interface / its prior, not a new
# relaxation; changing these requires a distinct prospective protocol.
PUSHER_COMPONENTS_MAX = 2
PUSHER_BBOX_MAX_PIXELS = 20


def config_bytes(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _failure(reason, config_sha):
    return {"schema": SCHEMA, "status": "abstain", "reason": reason,
            "candidates": [], "config_sha256": config_sha,
            "P1_admission": False, "physical_execution_authorized": False}


def _image(value):
    x = np.asarray(value)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 8:
        raise ValueError("RGB must be uint8 HxWx3")
    return x


def _component(image, color, min_pixels, *, pusher=False):
    mask = np.all(image == color, axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        raise ValueError("segmentation_missing_or_small")
    height, width = mask.shape
    if (xs == 0).any() or (ys == 0).any() or (xs == width - 1).any() or (ys == height - 1).any():
        raise ValueError("segmentation_clipped")
    if pusher and (int(xs.max() - xs.min() + 1) > PUSHER_BBOX_MAX_PIXELS
                   or int(ys.max() - ys.min() + 1) > PUSHER_BBOX_MAX_PIXELS):
        raise ValueError("pusher_bbox_too_large")
    visited = set()
    components = 0
    for y0, x0 in zip(ys, xs):
        seed = (int(y0), int(x0))
        if seed in visited:
            continue
        components += 1
        if components > (PUSHER_COMPONENTS_MAX if pusher else 1):
            raise ValueError("pusher_components_exceeded" if pusher else "segmentation_fragmented")
        pending = [seed]
        visited.add(seed)
        while pending:
            y, x = pending.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                point = (y + dy, x + dx)
                if (0 <= point[0] < height and 0 <= point[1] < width
                        and mask[point] and point not in visited):
                    visited.add(point)
                    pending.append(point)
    points = np.column_stack((xs, ys)).astype(np.float64)
    return points, points.mean(axis=0)


def _inside(center, radius, width, height):
    return (radius <= center[0] <= width - 1 - radius
            and radius <= center[1] <= height - 1 - radius)


def _segment_gap(a, b, points):
    delta = b - a
    length2 = float(delta @ delta)
    t = np.zeros(len(points)) if length2 == 0 else np.clip((points - a) @ delta / length2, 0, 1)
    return float(np.linalg.norm(points - (a + t[:, None] * delta), axis=1).min())


def _validate_config(config, expected_sha):
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("expected config sha256 required")
    try:
        sha = hashlib.sha256(config_bytes(config)).hexdigest()
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid configuration") from exc
    if sha != expected_sha:
        raise ValueError("config SHA mismatch")
    keys = {"schema", "current_action_pixels", "previous_action_pixels",
            "radius_pixels", "clearance_pixels", "steps", "max_action",
            "min_component_pixels", "max_candidates", "target_separation_pixels",
            "min_goal_displacement_pixels", "min_alignment", "model_id"}
    if not isinstance(config, dict) or set(config) != keys or config["schema"] != CONFIG_SCHEMA:
        raise ValueError("config schema")
    for key in ("current_action_pixels", "previous_action_pixels", "radius_pixels",
                "clearance_pixels", "max_action", "target_separation_pixels",
                "min_goal_displacement_pixels", "min_alignment"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("finite config " + key)
    if not (config["current_action_pixels"] > 0 and config["previous_action_pixels"] >= 0
            and config["radius_pixels"] > 0 and config["clearance_pixels"] > 0
            and 0 < config["max_action"] <= 1 and config["target_separation_pixels"] > 0
            and config["min_goal_displacement_pixels"] > 0 and 0 < config["min_alignment"] <= 1):
        raise ValueError("config bounds")
    if not (type(config["steps"]) is int and 1 <= config["steps"] <= 100
            and type(config["min_component_pixels"]) is int and config["min_component_pixels"] >= 1
            and type(config["max_candidates"]) is int and 2 <= config["max_candidates"] <= 32
            and isinstance(config["model_id"], str) and config["model_id"]):
        raise ValueError("config counts or model identity")
    return sha


def generate(current_rgb, goal_rgb, config, expected_config_sha256):
    """Generate bounded straight-line approaches or abstain.

    Assumes zero previous action at the start; configuration provenance and
    frozen-before-selection authority are the caller's responsibility.
    """
    sha = _validate_config(config, expected_config_sha256)
    try:
        current, goal = _image(current_rgb), _image(goal_rgb)
        if current.shape != goal.shape:
            raise ValueError("image_shape_mismatch")
        points, obj = _component(current, OBJECT, config["min_component_pixels"])
        _, start = _component(current, PUSHER, config["min_component_pixels"], pusher=True)
        _, goal_obj = _component(goal, OBJECT, config["min_component_pixels"])
    except (ValueError, TypeError) as exc:
        return _failure(str(exc), sha)
    height, width = current.shape[:2]
    desired = goal_obj - obj
    magnitude = float(np.linalg.norm(desired))
    if magnitude < config["min_goal_displacement_pixels"]:
        return _failure("goal_displacement_too_small", sha)
    desired /= magnitude
    radius = float(config["radius_pixels"])
    # Positive clearance is reserved for the whole predicted approach path.
    distance = radius + float(config["clearance_pixels"])
    if not _inside(start, radius, width, height):
        return _failure("pusher_disk_out_of_frame", sha)
    if np.linalg.norm(points - start, axis=1).min() < distance:
        return _failure("initial_clearance_unavailable", sha)

    # Only visible 4-neighbour boundary pixels are candidate support points.
    point_set = {(int(x), int(y)) for x, y in points}
    boundary = [p for p in point_set if any(
        (p[0] + dx, p[1] + dy) not in point_set
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    boundary.sort(key=lambda p: (float(np.dot(np.asarray(p) - obj, desired)), p[1], p[0]))
    steps = config["steps"]
    alpha, beta = config["current_action_pixels"], config["previous_action_pixels"]
    scale = steps * alpha + (steps - 1) * beta
    candidates = []
    for angle in ANGLES_DEG:
        theta = math.radians(angle)
        direction = np.array([math.cos(theta) * desired[0] - math.sin(theta) * desired[1],
                              math.sin(theta) * desired[0] + math.cos(theta) * desired[1]])
        if float(direction @ desired) + 1e-12 < config["min_alignment"]:
            continue
        for pixel in boundary:
            # Extra stand-off permits oblique approaches near a clipped
            # support disk while the full object clearance check remains strict.
            target = np.asarray(pixel, np.float64) - direction * (1.5 * distance)
            if not _inside(target, radius, width, height):
                continue
            if any(np.linalg.norm(target - np.asarray(c["target_xy"])) <
                   config["target_separation_pixels"] for c in candidates):
                continue
            command = np.asarray((target - start) / scale, dtype="<f4")
            if not np.isfinite(command).all() or np.max(np.abs(command)) > config["max_action"]:
                continue
            actions = np.tile(command, (steps, 1)).astype("<f4")
            centers = []
            pos, prev = start.copy(), np.zeros(2)
            good, min_gap = True, float("inf")
            for action in actions:
                next_pos = pos + alpha * action.astype(np.float64) + beta * prev
                gap = _segment_gap(pos, next_pos, points)
                min_gap = min(min_gap, gap)
                if not _inside(next_pos, radius, width, height) or gap < distance - 1e-5:
                    good = False
                    break
                centers.append(next_pos.copy())
                pos, prev = next_pos, action.astype(np.float64)
            if not good or np.linalg.norm(pos - target) > 1e-4:
                continue
            candidates.append({
                "support_pixel_xy": list(pixel), "target_xy": target.tolist(),
                "approach_direction_xy": direction.tolist(),
                "alignment_to_goal": float(direction @ desired),
                "actions": actions.tolist(),
                "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "predicted_centers_xy": np.asarray(centers).tolist(),
                "predicted_min_object_pixel_gap": min_gap,
            })
            if len(candidates) == config["max_candidates"]:
                break
        if len(candidates) == config["max_candidates"]:
            break
    if len(candidates) < 2:
        return _failure("fewer_than_two_safe_approaches", sha)
    return {
        "schema": SCHEMA, "status": "bounded_approach_witnesses",
        "config_sha256": sha, "candidates": candidates,
        "current_rgb_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
        "goal_rgb_sha256": hashlib.sha256(goal.tobytes()).hexdigest(),
        "pusher_start_xy": start.tolist(), "object_center_xy": obj.tolist(),
        "goal_center_xy": goal_obj.tolist(), "previous_action_assumed_zero": True,
        "rgb_geometry_only": True, "contact_or_physical_safety_proven": False,
        "P1_admission": False, "physical_execution_authorized": False,
    }
