import numpy as np
import pytest

from experiments.pc_exact_goal_relative_bridge_20261107_codex import make_chain


def test_waypoint_construction_and_action_bound():
    chain, point, steps = make_chain([0, 0, 0], [1, 0], 0.5)
    np.testing.assert_allclose(point, [.5, .5])
    assert chain.shape == (15, 2)
    assert np.max(np.abs(chain)) <= 1
    assert steps > 0


def test_invalid_or_degenerate_inputs_rejected():
    with pytest.raises(ValueError):
        make_chain([0, 0, 0], [0, 0], 1)
    with pytest.raises(ValueError):
        make_chain([0, 0, 0], [1, 0], float("nan"))
