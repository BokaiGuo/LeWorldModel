import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_rgb_boundary_contact_witness_20261102_exclusive as m,
)


def config():
    return {"schema": m.CONFIG_SCHEMA, "current_action_pixels": 8.0,
            "previous_action_pixels": 2.0, "radius_pixels": 4.0,
            "clearance_pixels": 1.0, "steps": 5, "max_action": 1.0,
            "min_component_pixels": 4, "max_candidates": 4,
            "target_separation_pixels": 2.0, "min_goal_displacement_pixels": 2.0,
            "min_alignment": 0.5, "model_id": "synthetic-frozen-model"}


def scenes():
    current = np.zeros((64, 64, 3), dtype=np.uint8)
    goal = np.zeros_like(current)
    current[27:37, 3:12] = m.OBJECT
    current[8:12, 8:12] = m.PUSHER
    goal[27:37, 30:39] = m.OBJECT
    return current, goal


def run(current, goal, cfg=None):
    cfg = cfg or config()
    return m.generate(current, goal, cfg, hashlib.sha256(m.config_bytes(cfg)).hexdigest())


def test_boundary_disk_overflow_gets_multiple_bounded_witnesses():
    current, goal = scenes()
    result = run(current, goal)
    assert result["status"] == "bounded_approach_witnesses"
    assert len(result["candidates"]) >= 2
    assert not result["contact_or_physical_safety_proven"]
    assert not result["P1_admission"]
    for row in result["candidates"]:
        actions = np.asarray(row["actions"], dtype="<f4")
        assert actions.shape == (5, 2) and np.max(np.abs(actions)) <= 1
        assert hashlib.sha256(actions.tobytes()).hexdigest() == row["actions_sha256"]
        assert np.allclose(row["predicted_centers_xy"][-1], row["target_xy"], atol=1e-4)
        assert row["predicted_min_object_pixel_gap"] >= 5 - 1e-5
        assert row["alignment_to_goal"] >= 0.5 - 1e-12
        x, y = row["target_xy"]
        assert 4 <= x <= 59 and 4 <= y <= 59


@pytest.mark.parametrize("which", ["current_object", "goal_object", "pusher"])
def test_clipped_segmentation_abstains(which):
    current, goal = scenes()
    image = goal if which == "goal_object" else current
    color = m.PUSHER if which == "pusher" else m.OBJECT
    image[0, 2:8] = color
    assert run(current, goal)["reason"] == "segmentation_clipped"


@pytest.mark.parametrize("which", ["current_object", "goal_object", "pusher"])
def test_fragmented_segmentation_abstains(which):
    current, goal = scenes()
    image = goal if which == "goal_object" else current
    color = m.PUSHER if which == "pusher" else m.OBJECT
    image[50, 50] = color
    assert run(current, goal)["reason"] == "segmentation_fragmented"


def test_unreachable_witness_abstains_and_config_tampering_rejected():
    current, goal = scenes()
    cfg = config()
    cfg["max_action"] = 0.001
    assert run(current, goal, cfg)["reason"] == "fewer_than_two_safe_approaches"
    with pytest.raises(ValueError, match="SHA"):
        m.generate(current, goal, cfg, "0" * 64)


def test_missing_or_malformed_rgb_abstains():
    current, goal = scenes()
    current[current == np.array(m.PUSHER)] = 0
    assert run(current, goal)["status"] == "abstain"
    current, goal = scenes()
    assert run(current.astype(float), goal)["status"] == "abstain"
