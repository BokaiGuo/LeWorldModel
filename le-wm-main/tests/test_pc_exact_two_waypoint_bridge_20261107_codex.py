import numpy as np
import pytest

from experiments import pc_exact_two_waypoint_bridge_20261107_codex as run


def test_waypoints_and_legal_actions():
    chain, points, needed = run.make_chain([0, 0, 0], [1, 0], .5)
    np.testing.assert_allclose(points, [[.25, .5], [.75, .5]])
    assert chain.shape == (15, 2)
    assert np.max(np.abs(chain)) <= 1
    assert needed >= 3


def test_zero_offset_reaches_goal_in_open_space():
    task = {"start": [-2., -2., 0.], "goal": [-1., -2.]}
    chain, _, needed = run.make_chain(task["start"], task["goal"], 0.)
    assert needed <= 15
    endpoint = run.prior.baseline.prior.exact.endpoints(task, chain[None])[0]
    np.testing.assert_allclose(endpoint, task["goal"], atol=1e-12)


@pytest.mark.parametrize("start,goal,offset", [([0, 0, 0], [0, 0], 0.), ([0, 0, 0], [1, 0], np.nan)])
def test_invalid_inputs(start, goal, offset):
    with pytest.raises(ValueError):
        run.make_chain(start, goal, offset)
