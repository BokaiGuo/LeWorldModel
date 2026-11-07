import numpy as np
import pytest

from experiments.pc_exact_primitive_cem_control_20261107_codex import chain_from_parameters, sample_round


def test_turn_drive_actions_and_truncation():
    chain, required = chain_from_parameters([0, 0, 0], [1, 0], [0, -1, 0, -1])
    assert chain.shape == (15, 2)
    assert required == 0
    np.testing.assert_array_equal(chain, np.zeros((15, 2)))
    chain, required = chain_from_parameters([0, 0, 0], [4, 0], [1, 1, -1, 1])
    assert required > 15
    assert np.max(np.abs(chain)) <= 1


def test_invalid_inputs_and_sampling():
    with pytest.raises(ValueError):
        chain_from_parameters([0, 0, 0], [0, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        chain_from_parameters([0, 0, 0], [1, 0], [0, 0, 0, 2])
    sample = sample_round(np.random.default_rng(3), np.ones(4), np.ones(4), 64)
    assert sample.shape == (64, 4)
    assert np.max(np.abs(sample)) <= 1


def test_reconstruct_a_geometric_waypoint_chain():
    from experiments.pc_exact_goal_relative_bridge_20261107_codex import make_chain

    start, goal = np.array([-.5, .2, .4]), np.array([.5, -.2])
    bridge, waypoint, _ = make_chain(start, goal, .35)
    direct = goal - start[:2]
    dist, bearing = np.linalg.norm(direct), np.arctan2(direct[1], direct[0])
    legs = (waypoint - start[:2], goal - waypoint)
    params = []
    for leg in legs:
        relative = (np.arctan2(leg[1], leg[0]) - bearing + np.pi) % (2 * np.pi) - np.pi
        params.extend((relative / (np.pi / 2), np.linalg.norm(leg) / dist - 1))
    decoded, _ = chain_from_parameters(start, goal, params)
    np.testing.assert_allclose(decoded, bridge, atol=1e-12, rtol=0)
