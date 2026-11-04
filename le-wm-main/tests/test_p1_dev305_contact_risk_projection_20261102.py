import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_risk_projection_20261102 import predict_screen


def test_projected_translation_and_outside_guard():
    jac = np.eye(192, 5)
    current = np.zeros(192)
    delta = np.zeros((4, 192))
    delta[1, 0] = -20
    boxes = {"object": [20, 20, 30, 30], "pusher": [100, 100, 110, 110]}
    rows = predict_screen(jac, delta, current, boxes)
    assert rows[0]["predicted_visible"]
    assert not rows[1]["predicted_visible"]
    assert rows[1]["estimated_translation_xy"]["object"][0] < -70
    assert rows[1]["estimated_translation_xy"]["pusher"] == [0, 0]


def test_zero_jacobian_fails_closed():
    with pytest.raises(ValueError):
        predict_screen(np.zeros((192, 5)), np.zeros((4, 192)), np.zeros(192),
                       {"object": [20, 20, 30, 30], "pusher": [40, 40, 50, 50]})
