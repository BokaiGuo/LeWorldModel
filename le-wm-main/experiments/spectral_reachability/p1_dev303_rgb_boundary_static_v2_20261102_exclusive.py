"""Read-only dev303 blind RGB static diagnostic for boundary witness v2."""
import json
from io import BytesIO

import numpy as np

from . import p1_dev303_rgb_boundary_static_20261102_exclusive as baseline
from . import p1_rgb_boundary_contact_witness_v2_20261102_exclusive as v2

NAME = "p1_dev303_rgb_boundary_static_v2_20261102_exclusive"


def evaluate(manifest_raw, blind_raw, motion_source_sha):
    # The existing checker verifies fixed hashes, pixels and zero action history;
    # it reads only the supplied blind material, and runs no physical system.
    checked = baseline.static_check(manifest_raw, blind_raw, motion_source_sha)
    with np.load(BytesIO(blind_raw), allow_pickle=False) as archive:
        current = archive["rgb_history"][-1].copy()
        goal = archive["goal_rgb"].copy()
    result = v2.generate(current, goal, baseline.CONFIG, checked["config_sha256"])
    return {
        "schema": NAME + ".v1",
        "classification": "development_only_not_independent",
        "witness_schema": v2.SCHEMA,
        "manifest_sha256": checked["manifest_sha256"],
        "blind_npz_sha256": checked["blind_npz_sha256"],
        "source_chain_reverified": False,
        "config": checked["config"],
        "config_sha256": checked["config_sha256"],
        "motion_source_sha256_observed_not_authority_verified":
            checked["motion_source_sha256_observed_not_authority_verified"],
        "pusher_guard": {"components_max": v2.PUSHER_COMPONENTS_MAX,
                         "bbox_max_pixels": v2.PUSHER_BBOX_MAX_PIXELS,
                         "connectivity": 4},
        "action_history": checked["action_history"],
        "goal_format": checked["goal_format"],
        "result": result, "candidate_count": len(result["candidates"]),
        "new_physics_calls": 0, "new_neural_calls": 0, "outcomes_read": False,
        "P1_admission": False, "independent_confirmation": False,
        "physical_execution_authorized": False,
    }


def main():
    result = evaluate(baseline.MANIFEST.read_bytes(), baseline.NPZ.read_bytes(),
                      baseline.sha(baseline.MOTION_SOURCE.read_bytes()))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
