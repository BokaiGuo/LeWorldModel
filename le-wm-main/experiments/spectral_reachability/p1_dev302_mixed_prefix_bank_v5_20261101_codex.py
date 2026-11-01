"""V5 RGB contact guard with a one-step-memory pusher motion model.

The no-memory V3/V4 guard was accurate for repeated direct actions but
misestimated sharp turns.  V5 uses coefficients fitted on development-only
non-contact RGB/native trajectories and keeps the V4 candidate actions fixed.
The resulting guard remains RGB-only at deployment and reads no simulator
state or outcome.
"""
from __future__ import annotations

import numpy as np

from . import p1_dev302_mixed_prefix_bank_v4_20261101_codex as v4
from . import p1_dev302_mixed_prefix_bank_20261101_codex as geometry

NATIVE_DELTA_PER_CURRENT_ACTION = 29.567
NATIVE_DELTA_PER_PREVIOUS_VELOCITY = 0.02993
NATIVE_VELOCITY_PER_PREVIOUS_ACTION = 374.1
PIXELS_PER_NATIVE = 224.0 / 512.0
CONTACT_CENTER_TO_OBJECT_PIXELS = 6.5


def _centers(start, actions):
    position = np.asarray(start, np.float64).copy()
    previous_action = np.zeros(2, np.float64)
    rows = []
    for action in np.asarray(actions, np.float64):
        native_delta = (
            NATIVE_DELTA_PER_CURRENT_ACTION * action
            + NATIVE_DELTA_PER_PREVIOUS_VELOCITY
            * NATIVE_VELOCITY_PER_PREVIOUS_ACTION
            * previous_action
        )
        position = position + native_delta * PIXELS_PER_NATIVE
        rows.append(position.copy())
        previous_action = action
    return np.asarray(rows)


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v4.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    image = geometry.rgb_source.rgb._image(current_rgb)
    object_mask, _ = geometry.rgb_source.rgb._mask(
        image, geometry.rgb_source.rgb.BLOCK, 8, single=False
    )
    py, px = np.nonzero(object_mask)
    object_pixels = np.stack((px, py), axis=1).astype(np.float64)
    start = np.asarray(result["geometry"]["pusher_xy"], np.float64)
    for row in result["candidates"]:
        centers = _centers(start, row["actions"])
        gaps = np.min(
            np.linalg.norm(centers[:, None, :] - object_pixels[None, :, :], axis=-1),
            axis=1,
        )
        minimum = float(gaps.min())
        row["rgb_dynamic_predicted_pusher_centers_xy"] = centers.tolist()
        row["rgb_dynamic_object_center_to_pixel_gap_by_step"] = gaps.tolist()
        row["rgb_dynamic_minimum_object_center_to_pixel_gap"] = minimum
        row["contact_feasible_prior"] = minimum <= CONTACT_CENTER_TO_OBJECT_PIXELS
    result["schema"] = "p1_dev302_mixed_prefix_bank_v5_20261101_codex.v1"
    result["contact_guard"] = {
        "kind": "rgb_development_calibrated_one_step_memory_motion_to_object_pixels",
        "native_delta_per_current_action": NATIVE_DELTA_PER_CURRENT_ACTION,
        "native_delta_per_previous_velocity": NATIVE_DELTA_PER_PREVIOUS_VELOCITY,
        "native_velocity_per_previous_action": NATIVE_VELOCITY_PER_PREVIOUS_ACTION,
        "pixels_per_native": PIXELS_PER_NATIVE,
        "contact_center_to_object_pixels": CONTACT_CENTER_TO_OBJECT_PIXELS,
        "candidate_actions_changed_from_v4": False,
        "simulator_state_used_at_deployment": False,
        "calibration_scope": "dev302 non-contact executed RGB/native trajectories",
        "prospective_status": "freeze before dev303 outcome",
    }
    return result
