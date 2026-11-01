"""V7 delayed-support path-diversity bank with identical predicted support endpoint."""
from __future__ import annotations

import hashlib
import numpy as np

from . import p1_dev302_mixed_prefix_bank_v6_20261101_codex as v6

PERTURBATIONS = (
    ("rgb_delayed_support_x12", 0, 1, 0, 0.20),
    ("rgb_delayed_support_y12", 0, 1, 1, 0.12),
    ("rgb_delayed_support_x23", 1, 2, 0, 0.20),
    ("rgb_delayed_support_y23", 1, 2, 1, 0.12),
)


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v6.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    base = v6.DELAYED_SUPPORT_ACTIONS
    for label, positive, negative, axis, amount in PERTURBATIONS:
        actions = base.copy()
        actions[positive, axis] += np.float32(amount)
        actions[negative, axis] -= np.float32(amount)
        result["candidates"].append(
            {
                "index": len(result["candidates"]),
                "label": label,
                "actions": actions.tolist(),
                "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
                "nearest_contact_projection_first2": None,
                "constructed_from_rgb_contact_geometry": True,
                "delayed_support_design": "endpoint-preserving path diversity",
            }
        )
    original = result["candidates"][:9]
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    reference = np.asarray(
        original[8]["rgb_dynamic_predicted_pusher_centers_xy"][-1], np.float64
    )
    for row in result["candidates"][9:]:
        centers = v6.v5._centers(start, row["actions"])
        require_close = float(np.linalg.norm(centers[-1] - reference))
        if require_close > 1e-4:
            raise ValueError("endpoint drift")
        row["rgb_dynamic_predicted_pusher_centers_xy"] = centers.tolist()
        for key in (
            "rgb_dynamic_final_object_center_to_pixel_gap",
            "rgb_dynamic_final_contact_direction_xy",
            "task_support_alignment",
            "contact_feasible_prior",
        ):
            row[key] = original[8][key]
    result["schema"] = "p1_dev302_mixed_prefix_bank_v7_20261101_codex.v1"
    result["contact_guard"] = {
        **result["contact_guard"],
        "kind": "rgb_task_support_with_endpoint_preserving_path_diversity",
        "path_variants": [item[0] for item in PERTURBATIONS],
    }
    return result
