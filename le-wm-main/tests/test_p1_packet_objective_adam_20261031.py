import torch

from experiments.spectral_reachability import p1_packet_objective_adam_20261031 as m


def test_packet_objective_uses_both_times_and_preserves_backprop(monkeypatch):
    def context(row, cp, device):
        return (torch.zeros(1, 3, 1, device=device), torch.zeros(1, 2, 10, device=device),
                torch.zeros(1, 1, device=device), torch.zeros(5, 2, device=device),
                torch.ones(5, 2, device=device))

    class Stub:
        def action_encoder(self, x):
            return x

        def predict(self, states, actions):
            return states + actions[:, -1:, :1]

    monkeypatch.setattr(m.source.adam.prior.base.bridge.cp5, "context", context)
    actions = torch.zeros(1, 2, 5, 2, requires_grad=True)
    with torch.no_grad():
        actions[0, 0, 0, 0] = 1
        actions[0, 1, 0, 0] = 2
    first, second = m.packet_losses(Stub(), {}, torch.tensor([0.]), actions)
    assert first.item() == 1
    assert second.item() == 9
    (0.5 * (first + second)).backward()
    assert actions.grad[0, 0, 0, 0].item() == 4
    assert actions.grad[0, 1, 0, 0].item() == 3
