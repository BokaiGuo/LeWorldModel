import pytest
import torch
from experiments.spectral_reachability import p1_suffix_exchange_q_20261101_main as x


def test_complete_cross_product_and_diagonal():
    bank = torch.arange(240, dtype=torch.float32).reshape(8, 3, 5, 2) / 256
    crossed = x.cross_bank(bank)
    for i in range(8):
        for j in range(8):
            assert torch.equal(crossed[8*i+j, :1], bank[i, :1])
            assert torch.equal(crossed[8*i+j, 1:], bank[j, 1:])
        assert torch.equal(crossed[9*i], bank[i])


def test_ties_preserve_own_tail():
    assert x.best_per_prefix([0.] * 64) == list(range(8))
    values = [1.] * 64
    values[8*3+7] = 0.
    assert x.best_per_prefix(values)[3] == 7
    with pytest.raises(ValueError):
        x.best_per_prefix([float("nan")] * 64)


@pytest.mark.parametrize("h", [1, 2, 3])
def test_exact_exchange_cost(monkeypatch, h):
    monkeypatch.setattr(x.parent.base, "CONFIG", dict(steps=6, planning_horizon_packets=h))
    for arm in x.parent.spec.ARMS:
        before, after = x.BASE_COST(arm), x.cost(arm)
        assert after["predict_calls"] - before["predict_calls"] == (64*h if h > 1 else 0)
        assert after["replay_calls"] - before["replay_calls"] == (64 if h > 1 else 0)
        assert after["backward_calls"] == before["backward_calls"]
