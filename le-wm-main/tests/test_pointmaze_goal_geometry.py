from __future__ import annotations

import numpy as np

from experiments.audit_pointmaze_goal_geometry import audit_goal_geometry, legacy_collision_geometry
from experiments.run_live_pointmaze_planning import GOAL_STATE


def test_legacy_geometry_exposes_world_offset_particle_radius_and_walls() -> None:
    geometry = legacy_collision_geometry()
    assert np.allclose(geometry["particle_body_world_offset"], [1.2, 1.2])
    assert geometry["particle_radius"] == 0.1
    assert len(geometry["walls"]) == 18


def test_legacy_goal_is_a_wall_sensitive_terminal_target() -> None:
    report = audit_goal_geometry(GOAL_STATE, success_radius=0.25, robust_clearance=0.05)
    assert report["target_is_collision_free"]
    assert not report["target_has_robust_clearance"]
    assert report["success_disk_intersects_collision_boundary"]
    assert report["interpretation"] == "legacy_target_near_or_at_wall_contact"


def test_audit_rejects_invalid_goal_vector() -> None:
    try:
        audit_goal_geometry(np.asarray([float("nan"), 0.0]))
    except ValueError as error:
        assert "finite [2]" in str(error)
    else:
        raise AssertionError("non-finite target was accepted")
