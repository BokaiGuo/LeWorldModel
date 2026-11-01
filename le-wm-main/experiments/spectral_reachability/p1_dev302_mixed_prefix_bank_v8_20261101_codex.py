"""V8 larger endpoint-preserving path variations for non-greedy support search."""
from __future__ import annotations

import hashlib
import numpy as np

from . import p1_dev302_mixed_prefix_bank_v7_20261101_codex as v7

LARGE = (
    ("rgb_delayed_x12_pos06", 0, 1, 0.60),
    ("rgb_delayed_x12_neg06", 0, 1, -0.60),
    ("rgb_delayed_x23_pos06", 1, 2, 0.60),
    ("rgb_delayed_x23_neg04", 1, 2, -0.40),
)


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v7.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    reference = np.asarray(
        result["candidates"][8]["rgb_dynamic_predicted_pusher_centers_xy"][-1]
    )
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    template = v7.v6.DELAYED_SUPPORT_ACTIONS
    source = result["candidates"][8]
    for label, positive, negative, amount in LARGE:
        actions = template.copy()
        actions[positive, 0] += np.float32(amount)
        actions[negative, 0] -= np.float32(amount)
        if np.max(np.abs(actions)) > 1:
            raise ValueError("large variation bounds")
        centers = v7.v6.v5._centers(start, actions)
        if np.linalg.norm(centers[-1] - reference) > 1e-4:
            raise ValueError("large variation endpoint")
        result["candidates"].append(
            {
                **{
                    key: source[key]
                    for key in (
                        "rgb_dynamic_final_object_center_to_pixel_gap",
                        "rgb_dynamic_final_contact_direction_xy",
                        "task_support_alignment",
                        "contact_feasible_prior",
                    )
                },
                "index": len(result["candidates"]),
                "label": label,
                "actions": actions.tolist(),
                "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
                "nearest_contact_projection_first2": None,
                "constructed_from_rgb_contact_geometry": True,
                "delayed_support_design": "large endpoint-preserving path diversity",
                "rgb_dynamic_predicted_pusher_centers_xy": centers.tolist(),
            }
        )
    result["schema"] = "p1_dev302_mixed_prefix_bank_v8_20261101_codex.v1"
    result["contact_guard"]["path_variants"] += [item[0] for item in LARGE]
    return result
