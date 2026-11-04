import numpy as np
import pytest

from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("mode", ["same", "incumbent", "zero"])
def test_paired_selection_and_budget(monkeypatch, step, mode):
    length = 15 - 5 * step
    first = np.zeros((length, 5))
    zero = first.copy()
    if mode != "same":
        first[0, 0] = .25
    else:
        # Different float64 representations can still produce identical dispatched float32 actions.
        first[0, 0] = np.float64(np.float32(.25)) + 1e-10
        zero[0, 0] = .25
    meter = Meter(1920)

    def fake_v6(strategy, arm, root, local_step, target, predict, proxy, nominals, carry):
        assert proxy.token_cap == 1920 - 2 * (length // 5)
        assert proxy.token_samples == 0
        return {"precomparison": {"chain": first.tolist()}, "comparison_chain": zero.tolist(),
                "selected": {"name": "old"}, "abstained": False}

    def evaluate(actions):
        assert np.asarray(actions).shape == (2, length, 5)
        return np.zeros((2, 2)), np.array([1., 2.] if mode != "zero" else [2., 1.])

    monkeypatch.setattr(v7.v6, "decide", fake_v6)
    result = v7.decide("response", "response", 0, step, [0, 0],
                       lambda x: meter.predict(np.asarray(x), evaluate), meter, np.zeros((2, 5, 5)))
    assert result["abstained"] is (mode != "incumbent")
    assert result["paired_comparison"]["same_actual_float32_actions"] is (mode == "same")
    assert meter.token_samples == 2 * (length // 5)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_real_v6_search_with_reserved_pair(step):
    meter = Meter(1920)

    def evaluate(actions):
        values = np.asarray(actions)
        position = np.stack((values.sum(axis=(1, 2)), (values ** 2).sum(axis=(1, 2))), axis=1)
        return position, np.linalg.norm(position - [.7, .4], axis=1)

    carry = None if step == 0 else np.zeros((15-5*step, 5))
    result = v7.decide("response", "response", 0, step, [.7, .4],
                       lambda x: meter.predict(np.asarray(x), evaluate), meter,
                       np.stack((np.zeros((5, 5)), np.full((5, 5), .2))), carry)
    assert meter.token_samples <= 1920
    assert np.asarray(result["selected"]["chain"]).shape == (15-5*step, 5)
    assert result["paired_comparison"]["reserved_token_samples"] == 2*(3-step)
