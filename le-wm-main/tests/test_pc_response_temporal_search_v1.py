import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_response_temporal_search_v1 as method


def test_interpolation_endpoints_and_cem_retains_incumbent():
    knots = np.array([[[0., 1.], [.2, .4], [.4, -.2], [.6, -.4]]])
    result = method.interpolate(knots, 10)
    np.testing.assert_allclose(result[0, 0], knots[0, 0])
    np.testing.assert_allclose(result[0, -1], knots[0, -1])
    score = lambda x: (np.zeros((len(x), 2)), np.ones(len(x)))
    base = np.zeros((10, 2))
    chain, best, history = method.cem(base, np.zeros((5, 2)), 13, np.random.default_rng(4), score, .5, 4)
    assert best == .5
    np.testing.assert_array_equal(chain, np.zeros((15, 2)))
    assert [item["size"] for item in history] == [12, 1]


@pytest.mark.parametrize("arm", method.ARMS)
def test_each_arm_obeys_budget_and_selected_is_scored(arm):
    meter = Meter(method.CAP)
    scored = []

    def predict(actions):
        actions = np.asarray(actions, dtype=float)

        def raw(batch):
            position = batch.mean(axis=1)
            score = np.linalg.norm(position - np.array([.3, -.2]), axis=1)
            if batch.shape[1] == 15:
                scored.extend(zip(batch.copy(), score.tolist(), strict=True))
            return position, score

        return meter.predict(actions, raw)

    result = method.decide(arm, 0, np.array([.3, -.2]), predict, meter)
    chain = np.asarray(result["selected"]["chain"])
    assert chain.shape == (15, 2)
    assert meter.token_samples <= method.CAP
    assert any(np.array_equal(chain, action) and score == result["selected"]["terminal_score"]
               for action, score in scored)
    assert result["selected"]["terminal_score"] == min(score for _, score in scored)
    if arm in ("response", "matched_random"):
        assert len(result["retained"]) == 4
        assert all(len(program["generations"]) >= 2 for program in result["programs"])
