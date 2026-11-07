import numpy as np
import pytest

from experiments.pc_osfal_layer_observation_20261107_codex import measure


def test_measure_distinguishes_view_and_action_and_physical_regret():
    features = np.zeros((4, 2, 4, 1, 1))
    features[:, :, 1, 0, 0] = 1
    features[:, :, 2, 0, 0] = 2
    features[:, 1, :, 0, 0] += 3
    positions = np.tile([[2., 0.], [1., 0.], [0., 0.]], (4, 1, 1))
    rows = measure(features, positions, np.zeros((4, 2)))
    assert rows[0]["view_gap_mse"] == 9
    assert rows[0]["action_gap_mse"] == 2
    assert rows[0]["views"][0]["physical_regret"] == 2
    assert rows[0]["views"][1]["physical_regret"] == 2


def test_reject_bad_dimensions():
    with pytest.raises(ValueError):
        measure(np.zeros((1,)), np.zeros((4, 3, 2)), np.zeros((4, 2)))
