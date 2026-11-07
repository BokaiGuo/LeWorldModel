import numpy as np
import pytest

from experiments.pc_pointmaze_goal25_packet_cem_20261107_codex import packet_samples, select_scored


def test_packet_sampling_is_bounded_and_temporally_tied():
    mean, std = np.zeros((5, 2), np.float32), np.full((5, 2), .5, np.float32)
    actions = np.repeat(packet_samples(np.random.default_rng(7), mean, std), 5, axis=1)
    assert actions.shape == (16, 25, 2)
    assert np.abs(actions).max() <= 1
    np.testing.assert_array_equal(actions.reshape(16, 5, 5, 2), np.broadcast_to(actions.reshape(16, 5, 5, 2)[:, :, :1], (16, 5, 5, 2)))
    np.testing.assert_array_equal(actions[0], np.zeros((25, 2)))


def test_budget_selection_cannot_read_later_candidates():
    costs = np.ones(160)
    costs[12], costs[150] = .2, .1
    assert select_scored(costs, 48) == 12
    assert select_scored(costs, 160) == 150
    with pytest.raises(ValueError):
        select_scored(costs, 49)
