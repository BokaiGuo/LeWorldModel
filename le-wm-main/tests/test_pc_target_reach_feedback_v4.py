import numpy as np
import pytest

from experiments.spectral_reachability import pc_target_reach_feedback_v4 as method


class Meter:
    token_cap = 1920

    def __init__(self, target):
        self.token_samples = 0
        self.target = target
        self.full = []

    def predict(self, actions):
        actions = np.asarray(actions)
        self.token_samples += len(actions) * actions.shape[1] // 5
        assert self.token_samples <= self.token_cap
        pos = actions.sum(axis=1) / 5
        scores = np.linalg.norm(pos - self.target, axis=1)
        return pos, scores


@pytest.mark.parametrize("step", [0, 1, 2])
def test_full_feedback_chain_respects_cap_and_action_bounds(step):
    target = np.asarray([.4, -.2])
    meter = Meter(target)
    carry = None if step == 0 else np.zeros((15 - 5 * step, 2))
    result = method.decide("response", 2, step, target, meter.predict, meter, carry)
    chain = np.asarray(result["selected"]["chain"])
    assert chain.shape == (15 - 5 * step, 2)
    assert np.abs(chain).max() <= 1
    assert meter.token_samples <= 1920
    assert len(result["refinement"]) > 1
    assert len(result["successor_responses"]) == (len(result["first_bank"]) if step < 2 else 0)
    if step < 2:
        assert len(result["retained"]) == 4
        assert result["structure"]["svd"] == 2 * (2 + len(result["first_bank"]))


def test_target_changes_mode_contributions():
    def predict(actions):
        pos = np.asarray(actions).sum(axis=1) / 5
        return pos, np.zeros(len(pos))

    def run(target):
        structure = {"svd_seconds": 0., "svd": 0, "fd_action_samples": 0}
        return method.response(np.zeros((5, 2)), [], np.asarray(target), predict, structure)[1]

    assert run([1, 0])["mode_contributions"] != run([0, 1])["mode_contributions"]
