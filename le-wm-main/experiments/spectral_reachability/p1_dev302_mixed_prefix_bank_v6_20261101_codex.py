"""V6 task-support contact bank with a delayed fifth-step support candidate."""
from __future__ import annotations

import hashlib

import numpy as np

from . import p1_dev302_mixed_prefix_bank_v5_20261101_codex as v5
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry

MIN_SUPPORT_ALIGNMENT = 0.5
DELAYED_SUPPORT_ACTIONS = np.asarray(
    [
        [-0.257, -1.0],
        [-0.134, -0.935],
        [-0.336, -0.96],
        [-1.0, -0.95],
        [-1.0, 0.595],
    ],
    np.float32,
)


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v5.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    image = geometry.rgb_source.rgb._image(current_rgb)
    object_mask, _ = geometry.rgb_source.rgb._mask(
        image, geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    desired = np.asarray(result["geometry"]["desired_object_direction"], np.float64)
    actions = DELAYED_SUPPORT_ACTIONS.copy()
    result["candidates"].append(
        {
            "index": len(result["candidates"]),
            "label": "rgb_delayed_task_support",
            "actions": actions.tolist(),
            "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
            "nearest_contact_projection_first2": None,
            "constructed_from_rgb_contact_geometry": True,
            "delayed_support_design": "clear through step4; target support-face contact at step5",
        }
    )
    for row in result["candidates"]:
        centers = v5._centers(start, row["actions"])
        final = centers[-1]
        deltas = object_pixels - final
        distances = np.linalg.norm(deltas, axis=1)
        nearest = int(np.argmin(distances))
        gap = float(distances[nearest])
        direction = deltas[nearest] / gap
        alignment = float(direction @ desired)
        row["rgb_dynamic_predicted_pusher_centers_xy"] = centers.tolist()
        row["rgb_dynamic_final_object_center_to_pixel_gap"] = gap
        row["rgb_dynamic_final_contact_direction_xy"] = direction.tolist()
        row["task_support_alignment"] = alignment
        row["contact_feasible_prior"] = (
            gap <= v5.CONTACT_CENTER_TO_OBJECT_PIXELS
            and alignment >= MIN_SUPPORT_ALIGNMENT
        )
    result["schema"] = "p1_dev302_mixed_prefix_bank_v6_20261101_codex.v1"
    result["candidate_order_fixed"] = True
    result["contact_guard"] = {
        "kind": "rgb_dynamic_final_contact_and_task_support_alignment",
        "contact_center_to_object_pixels": v5.CONTACT_CENTER_TO_OBJECT_PIXELS,
        "minimum_task_support_alignment": MIN_SUPPORT_ALIGNMENT,
        "delayed_support_actions": DELAYED_SUPPORT_ACTIONS.tolist(),
        "simulator_state_used_at_deployment": False,
        "prospective_status": "requires development holdout after dev302",
    }
    return result
