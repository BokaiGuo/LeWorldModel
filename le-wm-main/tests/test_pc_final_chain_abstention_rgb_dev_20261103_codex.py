import numpy as np
import pytest

from experiments import pc_final_chain_abstention_rgb_dev_20261103_codex as adapter


class Meter:
    token_samples = 0


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("source", ["joint_refined", "carry"])
@pytest.mark.parametrize("counter_score,abstained", [(2., False), (1., True), (0., True)])
def test_final_chain_comparison_is_metered_and_uses_actual_suffix(
        monkeypatch, step, source, counter_score, abstained):
    horizon = 15 - 5 * step
    chain = np.full((horizon, 2), .3)
    meter = Meter()

    def fake_decide(arm, root, actual_step, target, predict, reserved, carry):
        assert reserved.token_samples == horizon // 5
        meter.token_samples += 10
        return {"selected": {"name": source, "chain": chain.tolist(), "terminal_score": 1.},
                "selection_source": source, "abstained_reconfiguration": False}

    def predict(actions):
        meter.token_samples += len(actions) * actions.shape[1] // 5
        return np.zeros((len(actions), 2)), np.full(len(actions), counter_score)

    monkeypatch.setattr(adapter.adaptive, "decide", fake_decide)
    result = adapter.decide("adaptive", 0, step, np.zeros(2), predict, meter)
    comparison = np.asarray(result["final_prefix_comparison"]["comparison_chain"])
    np.testing.assert_array_equal(comparison[:5], 0.)
    np.testing.assert_array_equal(comparison[5:], chain[5:])
    assert result["abstained_reconfiguration"] is abstained
    np.testing.assert_array_equal(result["selected"]["chain"], comparison if abstained else chain)
    assert result["final_prefix_comparison"]["tokens_used_this_decision"] == 10 + horizon // 5


def test_unmetered_counterfactual_fails_closed(monkeypatch):
    monkeypatch.setattr(adapter.adaptive, "decide", lambda *args: {
        "selected": {"name": "carry", "chain": np.zeros((15, 2)).tolist(), "terminal_score": 1.}})
    with pytest.raises(ValueError, match="not fully metered"):
        adapter.decide("adaptive", 0, 0, np.zeros(2),
                       lambda x: (np.zeros((1, 2)), np.zeros(1)), Meter())
