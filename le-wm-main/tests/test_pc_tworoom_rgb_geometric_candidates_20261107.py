import numpy as np
import pytest

from experiments import pc_tworoom_rgb_geometric_candidates_20261107_codex as experiment


def images():
    with np.load(experiment.panel.PUBLIC, allow_pickle=False) as data:
        return data["current"][0].copy(), data["goal"][0].copy()


def test_rgb_geometry_matches_evaluation_positions():
    current, goal = images()
    start, target, wall, door = experiment.geometry(current, goal)
    with np.load(experiment.panel.TRUTH, allow_pickle=False) as truth:
        assert np.linalg.norm(start - truth["initial"][0]) < 1.
        assert np.linalg.norm(target - truth["target"][0]) < 1.
    assert 100 < wall < 124
    assert 35 <= door <= 63


@pytest.mark.parametrize("remaining", [5, 10, 15, 20, 25])
def test_candidate_bounds_and_determinism(remaining):
    current, goal = images()
    pool, parsed = experiment.candidate_pool(current, goal, None, remaining)
    repeated, _ = experiment.candidate_pool(current, goal, None, remaining)
    assert parsed is not None
    assert pool.shape == (16, remaining, 2)
    assert np.max(np.abs(pool)) <= 1
    np.testing.assert_array_equal(pool, repeated)
    np.testing.assert_array_equal(pool[0], 0)


def test_invalid_images_fallback_and_invalid_horizon_rejected():
    current, goal = images()
    pool, parsed = experiment.candidate_pool(np.zeros_like(current), goal, None, 5)
    assert parsed is None
    np.testing.assert_array_equal(pool, 0)
    with pytest.raises(ValueError):
        experiment.candidate_pool(current, goal, None, 7)
