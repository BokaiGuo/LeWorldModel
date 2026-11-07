import numpy as np
import pytest

from experiments.pc_pusht_rgb_rotation_tangent_20261107_codex import projected_cost


def test_removes_only_direction_component():
    pred = np.array([[3., 4.], [5., -4.]])
    score, abstain = projected_cost(pred, np.zeros(2), np.array([1., 0.]))
    np.testing.assert_allclose(score, [16., 16.])
    score2, _ = projected_cost(pred, np.zeros(2), np.array([-10., 0.]))
    np.testing.assert_allclose(score, score2)
    assert not abstain


def test_undefined_direction_reverts_to_original():
    pred = np.array([[3., 4.]])
    for direction in (np.zeros(2), np.array([np.nan, 1.])):
        score, abstain = projected_cost(pred, np.zeros(2), direction)
        np.testing.assert_allclose(score, [25.])
        assert abstain


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        projected_cost(np.array([[np.nan, 0.]]), np.zeros(2), np.ones(2))
