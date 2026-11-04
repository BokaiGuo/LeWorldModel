import numpy as np
import pytest

from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments.pc_equal_resource_dev_20261103_codex import Meter


def predictor(meter):
    def call(actions):
        def raw(batch):
            positions = batch.sum(axis=1)
            scores = np.linalg.norm(positions - np.array([0.2, -0.3]), axis=1)
            return positions, scores
        return meter.predict(np.asarray(actions), raw)
    return call


@pytest.mark.parametrize("arm", policy.ARMS)
@pytest.mark.parametrize("step", (0, 1, 2))
def test_decision_respects_budget_and_fixed_terminal(arm, step):
    horizon = 15 - 5 * step
    meter = Meter(policy.CAP)
    carry = None if step == 0 else np.full((horizon, 2), 0.01)
    result = policy.decide(arm, 0, step, np.array([.2, -.3]), predictor(meter), meter, carry)
    assert np.asarray(result["selected"]["chain"]).shape == (horizon, 2)
    assert np.max(np.abs(result["selected"]["chain"])) <= 1
    assert 0 <= policy.CAP - meter.token_samples < horizon // 5
    if carry is not None:
        assert result["selected"]["terminal_score"] <= result["carry_score"]
    if arm in ("svd", "ridge", "greedy"):
        assert result["spectral_counts"]["prediction_samples"] == 42
    if result["programs"]:
        assert result["selected"]["terminal_score"] <= min(
            x["terminal_score"] for x in result["programs"]) + 1e-12


def test_refinement_preserves_incumbent_and_bind_carry():
    chain = np.zeros((10, 2))
    best, score = policy.refine(chain, 0., 15, np.random.default_rng(1),
                                lambda actions: np.ones(len(actions)))
    np.testing.assert_array_equal(best, chain)
    assert score == 0
    meter = Meter(policy.CAP)
    with pytest.raises(ValueError, match="tail"):
        policy.decide("cem", 0, 1, np.zeros(2), predictor(meter), meter)
