import numpy as np

from experiments import pc_tworoom_rgb_geometric_candidates_v2_20261107_codex as v2


def test_near_door_no_backward_waypoint(monkeypatch):
    monkeypatch.setattr(v2.v1, "geometry", lambda current, goal: (
        np.array([106., 49.]), np.array([140., 80.]), 112., 49.))
    pool, _ = v2.candidate_pool(None, None, None, 25)
    assert pool.shape == (16, 25, 2)
    assert pool[1, 0, 0] > 0
    assert np.max(np.abs(pool)) <= 1


def test_far_start_still_moves_to_safe_waypoint(monkeypatch):
    monkeypatch.setattr(v2.v1, "geometry", lambda current, goal: (
        np.array([75., 85.]), np.array([140., 80.]), 112., 49.))
    pool, _ = v2.candidate_pool(None, None, None, 25)
    assert pool[1, 0, 0] > 0
    assert pool[1, 0, 1] < 0
