import torch
import pytest
from experiments.spectral_reachability import p1_horizon_q_20261101_main as h


@pytest.mark.parametrize("packets", [1, 2, 3])
def test_rollout_consumes_only_requested_packets(packets):
    class Fixture:
        calls = 0

        def action_encode(self, context, phase):
            return context[:, -1:].sum(-1, keepdim=True)

        def predict(self, states, actions, phase):
            self.calls += 1
            return states + actions

    leaf = Fixture()
    actions = torch.ones((1, 3, 5, 2), requires_grad=True)
    result = h.rollout(leaf, torch.zeros((1, 3, 192)),
                       torch.zeros((1, 2, 10)), actions, torch.zeros(2), torch.ones(2), packets)
    result.sum().backward()
    assert leaf.calls == packets
    assert torch.count_nonzero(actions.grad[:, packets:]) == 0
    assert torch.count_nonzero(actions.grad[:, :packets]) == packets * 10


@pytest.mark.parametrize("packets", [1, 2, 3])
def test_cost_matches_explicit_packet_count(monkeypatch, packets):
    monkeypatch.setattr(h.base, "CONFIG", dict(h.base.CONFIG, steps=6, planning_horizon_packets=packets))
    for arm in h.spec.ARMS:
        value = h.cost(arm)
        expected = 8 * 7 * packets + packets * int(arm == "structured") + 16 * int(arm == "immediate_progress")
        assert value["predict_calls"] == expected
        assert value["backward_calls"] == 48 + 192 * int(arm == "structured")
