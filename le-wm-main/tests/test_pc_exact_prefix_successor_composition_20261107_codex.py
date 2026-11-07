import numpy as np
import pytest

from experiments.pc_exact_prefix_successor_composition_20261107_codex import (
    rollout_n, suffix_from_successor,
)
from experiments.pc_exact_goal_relative_bridge_20261107_codex import make_chain, baseline


def test_rollout_matches_existing_backend_with_padding():
    task = {"start": [-.5, -.15, .2], "goal": [.5, .35]}
    chains = np.random.default_rng(23).uniform(-1, 1, (4, 15, 2))
    actual, _ = rollout_n(task["start"], chains)
    expected = baseline.prior.exact.endpoints(task, chains)
    np.testing.assert_allclose(actual[:, -1, :2], expected, atol=1e-12, rtol=0)


def test_composition_tail_uses_actual_successor_and_is_bounded():
    start, goal = [-.5, -.15, 0.], [.5, .35]
    chain, waypoint, _ = make_chain(start, goal, .5)
    left, _ = suffix_from_successor(np.array([-.5, -.15, 0.]), waypoint, goal, False)
    moved, _ = suffix_from_successor(np.array([-.4, -.15, 0.]), waypoint, goal, False)
    assert left.shape == moved.shape == (10, 2)
    assert np.max(np.abs(left)) <= 1
    assert not np.array_equal(left, moved)
    assert chain.shape == (15, 2)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        rollout_n([0., 0., 0.], np.zeros((1, 5, 3)))
    with pytest.raises(ValueError):
        rollout_n([0., 0., 0.], np.full((1, 5, 2), 1.1))
    with pytest.raises(ValueError):
        suffix_from_successor([float("nan"), 0., 0.], [1., 0.], [2., 0.], False)
