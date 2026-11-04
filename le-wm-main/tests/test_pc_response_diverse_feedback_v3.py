import numpy as np

from experiments.spectral_reachability import pc_response_diverse_feedback_v1 as v1
from experiments.spectral_reachability import pc_response_diverse_feedback_v3 as v3


class TokenMeter:
    def __init__(self):
        self.token_samples = 0


def predictor(meter):
    probe = v1.core.prior.NOMINALS[0].copy()
    probe[0, 0] += .05
    trace = []

    def predict(actions):
        actions = np.asarray(actions)
        meter.token_samples += len(actions) * actions.shape[1] // 5
        scores = np.where(np.max(np.abs(actions - probe), axis=(1, 2)) < 1e-10, 0., 1.)
        trace.append((actions.copy(), scores.copy()))
        return actions[:, 0, :], scores

    return predict, trace, probe


def test_v3_preserves_search_and_selects_scored_probe():
    meters = [TokenMeter(), TokenMeter()]
    old_predict, old_trace, probe = predictor(meters[0])
    new_predict, new_trace, _ = predictor(meters[1])
    old = v1.decide("response_diverse", 0, 2, [0, 0], old_predict, meters[0], np.zeros((5, 2)))
    new = v3.decide("response_diverse", 0, 2, [0, 0], new_predict, meters[1], np.zeros((5, 2)))
    assert len(old_trace) == len(new_trace)
    for (old_actions, old_scores), (new_actions, new_scores) in zip(old_trace[:-1], new_trace[:-1], strict=True):
        np.testing.assert_array_equal(old_actions, new_actions)
        np.testing.assert_array_equal(old_scores, new_scores)
    assert old["selected"]["terminal_score"] == 1
    assert new["selected"]["terminal_score"] == 0
    np.testing.assert_array_equal(new["selected"]["chain"], probe)
    assert meters[0].token_samples == meters[1].token_samples == 480
