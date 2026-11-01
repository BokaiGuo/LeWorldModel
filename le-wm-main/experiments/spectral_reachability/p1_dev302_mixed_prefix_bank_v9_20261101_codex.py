"""V9 bounded repair of the V8 task-support candidate bank."""
from __future__ import annotations

import hashlib

import numpy as np

from . import p1_dev302_mixed_prefix_bank_v8_20261101_codex as v8


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v8.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    image = v8.v7.v6.geometry.rgb_source.rgb._image(current_rgb)
    object_mask, _ = v8.v7.v6.geometry.rgb_source.rgb._mask(
        image, v8.v7.v6.geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    desired = np.asarray(result["geometry"]["desired_object_direction"], np.float64)
    for row in result["candidates"]:
        original = np.asarray(row["actions"], np.float32)
        actions = np.clip(original, -1, 1).astype(np.float32)
        centers = v8.v7.v6.v5._centers(start, actions)
        all_deltas = object_pixels[None, :, :] - centers[:, None, :]
        all_gaps = np.linalg.norm(all_deltas, axis=-1)
        final_nearest = int(np.argmin(all_gaps[-1]))
        final_gap = float(all_gaps[-1, final_nearest])
        final_direction = all_deltas[-1, final_nearest] / final_gap
        alignment = float(final_direction @ desired)
        row["actions"] = actions.tolist()
        row["actions_sha256"] = hashlib.sha256(
            actions.astype("<f4", copy=False).tobytes()
        ).hexdigest()
        row["rgb_dynamic_predicted_pusher_centers_xy"] = centers.tolist()
        row["rgb_dynamic_object_center_to_pixel_gap_by_step"] = np.min(
            all_gaps, axis=1
        ).tolist()
        row["rgb_dynamic_minimum_object_center_to_pixel_gap"] = float(
            np.min(all_gaps)
        )
        row["rgb_dynamic_final_object_center_to_pixel_gap"] = final_gap
        row["rgb_dynamic_final_contact_direction_xy"] = final_direction.tolist()
        row["task_support_alignment"] = alignment
        row["contact_feasible_prior"] = (
            final_gap <= v8.v7.v6.v5.CONTACT_CENTER_TO_OBJECT_PIXELS
            and alignment >= v8.v7.v6.MIN_SUPPORT_ALIGNMENT
        )
        row["action_bound_repaired_from_v8"] = bool(
            not np.array_equal(original, actions)
        )
    result["schema"] = "p1_dev302_mixed_prefix_bank_v9_20261101_codex.v1"
    result["contact_guard"] = {
        **result["contact_guard"],
        "candidate_action_bounds": [-1.0, 1.0],
        "v8_out_of_bounds_candidates_repaired": [10, 12],
        "all_dynamic_geometry_recomputed_after_clipping": True,
        "prospective_status": "development repair; requires new frozen confirmation",
    }
    return result
