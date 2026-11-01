"""Outcome-free, label-agnostic RGB task-greedy scorer for P1."""
from __future__ import annotations

import hashlib

import numpy as np

from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry
from . import p1_dev302_mixed_prefix_bank_v5_20261101_codex as dynamics


FORBIDDEN_FIELDS = {
    "reward",
    "outcome",
    "native_success",
    "n_contacts",
    "transitions",
    "final_metrics",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _object_pixels(pixels):
    image = geometry.rgb_source.rgb._image(pixels)
    mask, obj = geometry.rgb_source.rgb._mask(
        image, geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(mask)
    points = np.stack((px, py), axis=1).astype(np.float64)
    require(
        len(points) >= 8
        and np.isfinite(points).all()
        and not (
            np.any(px == 0)
            or np.any(py == 0)
            or np.any(px == image.shape[1] - 1)
            or np.any(py == image.shape[0] - 1)
        ),
        "object mask",
    )
    return points, np.asarray(obj["centroid_xy"], np.float64)


def score_candidates(protocol, current_rgb, goal_rgb):
    bank = protocol["prefix_bank"]
    current_points, current_center = _object_pixels(current_rgb)
    _goal_points, goal_center = _object_pixels(goal_rgb)
    desired = goal_center - current_center
    norm = float(np.linalg.norm(desired))
    require(np.isfinite(norm) and norm > 1.0, "goal displacement")
    desired /= norm
    start = np.asarray(bank["geometry"]["pusher_xy"], np.float64)
    threshold = float(bank["contact_guard"]["contact_center_to_object_pixels"])
    minimum_alignment = float(bank["contact_guard"]["minimum_task_support_alignment"])
    require(
        start.shape == (2,)
        and np.isfinite(start).all()
        and threshold == dynamics.CONTACT_CENTER_TO_OBJECT_PIXELS
        and minimum_alignment == 0.5,
        "frozen support geometry",
    )
    rows, indices = [], set()
    for source in bank["candidates"]:
        require(not (FORBIDDEN_FIELDS & set(source)), "physical outcome field forbidden")
        index = source.get("index")
        require(
            type(index) is int and index >= 0 and index not in indices,
            "unique candidate index",
        )
        indices.add(index)
        actions = np.asarray(source.get("actions"), np.float32)
        require(
            actions.shape == (5, 2)
            and np.isfinite(actions).all()
            and float(np.max(np.abs(actions))) <= 1.0
            and hashlib.sha256(actions.astype("<f4", copy=False).tobytes()).hexdigest()
            == source.get("actions_sha256"),
            "actions SHA and bounds",
        )
        centers = dynamics._centers(start, actions)
        saved = np.asarray(
            source.get("rgb_dynamic_predicted_pusher_centers_xy"), np.float64
        )
        require(
            saved.shape == centers.shape == (5, 2)
            and np.allclose(saved, centers, atol=1e-9, rtol=0),
            "saved center recomputation",
        )
        opportunity, deficit = [], []
        for center in centers:
            deltas = current_points - center
            gaps = np.linalg.norm(deltas, axis=1)
            nearest = int(np.argmin(gaps))
            gap = float(gaps[nearest])
            require(np.isfinite(gap) and gap > 0, "support gap")
            alignment = float((deltas[nearest] / gap) @ desired)
            require(np.isfinite(alignment), "support alignment")
            opportunity.append(gap <= threshold and alignment >= minimum_alignment)
            deficit.append(
                float(
                    np.hypot(
                        max(gap - threshold, 0.0) / threshold,
                        max(minimum_alignment - alignment, 0.0)
                        / (1.0 + minimum_alignment),
                    )
                )
            )
        first = next(
            (step for step, present in enumerate(opportunity, 1) if present),
            len(opportunity) + 1,
        )
        rows.append(
            {
                "index": index,
                "opportunity_by_step": opportunity,
                "support_deficit_by_step": deficit,
                "score_key": [
                    -int(sum(opportunity)),
                    first,
                    float(np.mean(deficit)),
                    float(deficit[-1]),
                    index,
                ],
            }
        )
    require(rows, "nonempty candidate bank")
    return rows


def task_greedy_index(protocol, current_rgb, goal_rgb):
    return min(
        score_candidates(protocol, current_rgb, goal_rgb),
        key=lambda row: row["score_key"],
    )["index"]
