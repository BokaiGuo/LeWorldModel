import numpy as np
import pytest

from experiments.spectral_reachability import pc_response_diverse_feedback_v1 as v1
from experiments.spectral_reachability import pc_response_diverse_feedback_v2 as v2


class TokenMeter:
    def __init__(self):
        self.token_samples = 0


def make_predictor(meter):
    nominal = v1.core.prior.NOMINALS[0].copy()
    probe = nominal.copy()
    probe[0, 0] += .05
    recorded = []

    def predict(actions):
        actions = np.asarray(actions)
        meter.token_samples += len(actions) * actions.shape[1] // 5
        positions = actions[:, 0, :]
        scores = np.where(np.max(np.abs(actions - probe), axis=(1, 2)) < 1e-10, 0., 1.)
        recorded.extend((action.copy(), float(score)) for action, score in zip(actions, scores, strict=True))
        return positions, scores

    return predict, recorded, probe


def test_final_horizon_retains_best_response_probe():
    old_meter, new_meter = TokenMeter(), TokenMeter()
    old_predict, old_trace, probe = make_predictor(old_meter)
    new_predict, new_trace, _ = make_predictor(new_meter)
    old = v1.decide("response_diverse", 0, 2, [0, 0], old_predict, old_meter, np.zeros((5, 2)))
    new = v2.decide("response_diverse", 0, 2, [0, 0], new_predict, new_meter, np.zeros((5, 2)))
    assert min(score for _, score in old_trace) == 0
    assert old["selected"]["terminal_score"] == 1
    assert new["selected"]["terminal_score"] == min(score for _, score in new_trace) == 0
    np.testing.assert_array_equal(new["selected"]["chain"], probe)
    assert old_meter.token_samples == new_meter.token_samples == 480
    assert all(np.max(np.abs(action)) <= 1 for action, _ in new_trace)


def test_v2_rejects_invalid_carry_and_nonempty_meter():
    meter = TokenMeter()
    predict, _, _ = make_predictor(meter)
    with pytest.raises(ValueError):
        v2.decide("response_diverse", 0, 2, [0, 0], predict, meter, np.zeros((10, 2)))
    meter.token_samples = 1
    with pytest.raises(ValueError):
        v2.decide("response_diverse", 0, 2, [0, 0], predict, meter, np.zeros((5, 2)))
