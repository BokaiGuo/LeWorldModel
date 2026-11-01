import pytest
import torch

from experiments.spectral_reachability import p1_carry_host_20261101_main as m


def test_node_costs():
    assert m.node_extra(0) == m.node_extra(5) == {}
    assert m.node_extra(10)["predict_calls"] == 18
    assert m.node_extra(15)["predict_calls"] == 9
    assert m.node_extra(15)["replay_calls"] == 9
    with pytest.raises(ValueError):
        m.node_extra(7)


@pytest.mark.parametrize("horizon", [1, 2])
def test_extract_exact_plan(horizon):
    bank = torch.zeros(9, horizon, 5, 2)
    bank[-1] = .3
    result = dict(carry_bank=bank, selected_raw=bank[-1], selected_index=8)
    assert torch.equal(m.first_packet(result), bank[-1, 0])
    result["selected_raw"] = torch.ones_like(bank[-1])
    with pytest.raises(ValueError):
        m.first_packet(result)
