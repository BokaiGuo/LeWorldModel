"""V2 contact-feasibility guard for the fixed dev302 mixed prefix bank."""
from __future__ import annotations

import numpy as np

from . import p1_dev302_mixed_prefix_bank_20261101_codex as v1


def build(current_rgb, goal_rgb, spectral_prefix, gn_prefix):
    result = v1.build(current_rgb, goal_rgb, spectral_prefix, gn_prefix)
    direction = np.asarray(result["geometry"]["nearest_direction"], np.float64)
    for row in result["candidates"]:
        actions = np.asarray(row["actions"], np.float64)
        total = actions.sum(axis=0)
        alignment = total * direction
        active = np.abs(direction) >= 0.2
        componentwise = bool(np.all(alignment[active] > 0.0))
        if row["constructed_from_rgb_contact_geometry"]:
            feasible = True
        elif row["label"] == "zero":
            feasible = False
        else:
            feasible = bool(componentwise and float(total @ direction) >= 0.05)
        row["cumulative_action_xy"] = total.tolist()
        row["componentwise_contact_alignment_xy"] = alignment.tolist()
        row["componentwise_contact_progress"] = componentwise
        row["contact_feasible_prior"] = feasible
    result["schema"] = "p1_dev302_mixed_prefix_bank_v2_20261101_codex.v1"
    result["contact_guard"] = (
        "RGB-constructed contact candidates admitted; other candidates require "
        "positive cumulative progress on every nearest-contact axis with |direction|>=0.2"
    )
    return result
