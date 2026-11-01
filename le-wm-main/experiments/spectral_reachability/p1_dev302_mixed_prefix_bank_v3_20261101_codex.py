"""V3 RGB-calibrated contact-path guard for the dev302 mixed prefix bank."""
from __future__ import annotations

import numpy as np

from . import p1_dev302_mixed_prefix_bank_v2_20261101_codex as v2
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry

# Frozen from RGB-only development trajectories.  It is not a simulator-state
# conversion and must be revalidated prospectively on a new development root.
PIXELS_PER_CUMULATIVE_RAW = 17.0
CONTACT_CENTER_TO_OBJECT_PIXELS = 8.0


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v2.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    image = geometry.rgb_source.rgb._image(current_rgb)
    object_mask, _ = geometry.rgb_source.rgb._mask(
        image, geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    for row in result["candidates"]:
        actions = np.asarray(row["actions"], np.float64)
        centers = start + np.cumsum(actions, axis=0) * PIXELS_PER_CUMULATIVE_RAW
        gaps = np.min(
            np.linalg.norm(centers[:, None, :] - object_pixels[None, :, :], axis=-1),
            axis=1,
        )
        minimum = float(gaps.min())
        row["rgb_calibrated_predicted_pusher_centers_xy"] = centers.tolist()
        row["rgb_calibrated_object_boundary_gap_by_step"] = gaps.tolist()
        row["rgb_calibrated_minimum_object_boundary_gap"] = minimum
        row["contact_feasible_prior"] = minimum <= CONTACT_CENTER_TO_OBJECT_PIXELS
    result["schema"] = "p1_dev302_mixed_prefix_bank_v3_20261101_codex.v1"
    result["contact_guard"] = {
        "kind": "rgb_calibrated_action_path_to_visible_object_boundary",
        "pixels_per_cumulative_raw": PIXELS_PER_CUMULATIVE_RAW,
        "contact_center_to_object_pixels": CONTACT_CENTER_TO_OBJECT_PIXELS,
        "calibration": "historical RGB pusher displacement only; no simulator state",
        "prospective_status": "requires new development holdout",
    }
    return result
