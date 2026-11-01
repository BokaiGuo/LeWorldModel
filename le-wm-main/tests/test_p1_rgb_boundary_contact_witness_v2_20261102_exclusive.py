import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_rgb_boundary_contact_witness_v2_20261102_exclusive as m,
    p1_rgb_chunk_approach_interface_20261102_exclusive as chunk,
)


def config():
    return {"schema": m.CONFIG_SCHEMA, "current_action_pixels": 8.0,
            "previous_action_pixels": 2.0, "radius_pixels": 4.0,
            "clearance_pixels": 1.0, "steps": 5, "max_action": 1.0,
            "min_component_pixels": 4, "max_candidates": 4,
            "target_separation_pixels": 2.0, "min_goal_displacement_pixels": 2.0,
            "min_alignment": 0.5, "model_id": "fixture"}


def scene():
    current = np.zeros((64, 64, 3), dtype=np.uint8)
    goal = np.zeros_like(current)
    current[27:37, 3:12] = m.OBJECT
    current[8:10, 8:12] = m.PUSHER
    current[11:13, 8:12] = m.PUSHER
    goal[27:37, 30:39] = m.OBJECT
    return current, goal


def run(current, goal):
    cfg = config()
    return m.generate(current, goal, cfg, hashlib.sha256(m.config_bytes(cfg)).hexdigest())


def test_pusher_guard_limits_match_chunk_interface():
    assert m.PUSHER_COMPONENTS_MAX == chunk.prior.MOVING_PUSHER_COMPONENTS_MAX == 2
    assert m.PUSHER_BBOX_MAX_PIXELS == chunk.prior.MOVING_PUSHER_BBOX_MAX_PIXELS == 20


def test_two_nearby_components_can_produce_bounded_candidates():
    result = run(*scene())
    assert result["status"] == "bounded_approach_witnesses"
    assert len(result["candidates"]) >= 2
    for candidate in result["candidates"]:
        a = np.asarray(candidate["actions"], dtype="<f4")
        assert np.max(np.abs(a)) <= 1
        assert hashlib.sha256(a.tobytes()).hexdigest() == candidate["actions_sha256"]


def test_malicious_remote_fragment_cannot_shift_centroid():
    current, goal = scene()
    current[50, 50] = m.PUSHER
    result = run(current, goal)
    assert result["status"] == "abstain"
    assert result["reason"] == "pusher_bbox_too_large"


def test_three_nearby_components_rejected():
    current, goal = scene()
    current[15, 8] = m.PUSHER
    assert run(current, goal)["reason"] == "pusher_components_exceeded"


def test_large_contiguous_bbox_rejected():
    current, goal = scene()
    current[8:10, 8:29] = m.PUSHER
    assert run(current, goal)["reason"] == "pusher_bbox_too_large"


@pytest.mark.parametrize("target,color", [
    ("current", m.PUSHER), ("current", m.OBJECT), ("goal", m.OBJECT),
])
def test_clipped_masks_rejected(target, color):
    current, goal = scene()
    (current if target == "current" else goal)[0, 5] = color
    assert run(current, goal)["reason"] == "segmentation_clipped"


@pytest.mark.parametrize("target", ["current", "goal"])
def test_object_fragmentation_still_rejected(target):
    current, goal = scene()
    (current if target == "current" else goal)[50, 50] = m.OBJECT
    assert run(current, goal)["reason"] == "segmentation_fragmented"
