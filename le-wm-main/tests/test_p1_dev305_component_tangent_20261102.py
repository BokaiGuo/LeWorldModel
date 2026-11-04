import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_component_tangent_bank_20261102 import component_scores


def test_component_sum_and_batch_shape():
    jacobian = np.eye(6, 5)
    delta = np.arange(24, dtype=float).reshape(2, 2, 6)
    scores, ridge, singular = component_scores(delta, jacobian)
    assert ridge == pytest.approx(.01)
    assert singular.shape == (5,)
    assert scores["object"].shape == (2, 2)
    np.testing.assert_allclose(scores["joint"], scores["object"] + scores["pusher"])
    np.testing.assert_allclose(scores["object"], (delta[..., :3] ** 2).sum(-1) / 1.01 ** 2)


def test_rejects_invalid_inputs():
    with pytest.raises(Exception):
        component_scores(np.zeros(6), np.zeros((6, 5)))
    with pytest.raises(Exception):
        component_scores(np.zeros(5), np.eye(6, 5))
    with pytest.raises(Exception):
        component_scores(np.full(6, np.nan), np.eye(6, 5))
