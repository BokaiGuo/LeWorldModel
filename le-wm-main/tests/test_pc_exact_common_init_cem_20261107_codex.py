import numpy as np
import pytest

from experiments.pc_exact_common_init_cem_20261107_codex import waypoint_params
from experiments.pc_exact_goal_relative_bridge_20261107_codex import make_chain
from experiments.pc_exact_primitive_cem_control_20261107_codex import chain_from_parameters


@pytest.mark.parametrize("offset", [-1.9, -0.2, 0, 0.2, 1.9])
def test_common_chain_is_identical(offset):
    start, goal = [-.5, .2, .3], [.5, -.2]
    chain, waypoint, _ = make_chain(start, goal, offset)
    actual, _ = chain_from_parameters(start, goal, waypoint_params(start, goal, waypoint))
    np.testing.assert_allclose(chain, actual, rtol=0, atol=1e-12)


def test_degenerate_goal_rejected():
    with pytest.raises(ValueError):
        waypoint_params([0, 0, 0], [0, 0], [1, 0])
