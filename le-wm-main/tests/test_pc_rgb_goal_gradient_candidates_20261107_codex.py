import numpy as np
import pytest

from experiments.pc_rgb_goal_gradient_candidates_20261107_codex import make_candidates


def test_gradient_direction_and_bounded_candidates():
    raw = np.array([[.7, -.7], [0., .2]])
    gradient = np.array([[2., -1.], [.5, 0.]])
    actions, direction = make_candidates(raw, gradient)
    assert actions.shape == (5, 2, 2)
    np.testing.assert_allclose(direction, -gradient / 2)
    np.testing.assert_allclose(actions[1], raw + .15 * direction, atol=1e-7)
    np.testing.assert_allclose(actions[3], raw + .30 * direction, atol=1e-7)
    assert np.max(np.abs(actions)) <= 1.0000001


@pytest.mark.parametrize("raw,gradient", [
    (np.zeros((2, 2)), np.zeros((2, 2))),
    (np.ones((2, 2)), np.ones((2, 2))),
    (np.zeros((2, 2)), np.full((2, 2), np.nan)),
    (np.zeros((2, 2)), np.ones((3, 2))),
])
def test_invalid_input_rejected(raw, gradient):
    with pytest.raises(ValueError):
        make_candidates(raw, gradient)
