import pytest
import torch

from experiments.spectral_reachability import reacher_pc_finite_response_screen as module

shortlist = module.shortlist


def test_controls_and_deterministic_farthest_first():
    signatures = torch.tensor([[0.0], [0.0], [1.0], [-2.0], [1.0], [0.5]])
    assert shortlist(signatures, keep=3) == [0, 1, 3, 2, 5]


def test_ties_and_invalid_values_fail_closed():
    assert shortlist(torch.tensor([[0.0], [0.0], [1.0], [-1.0]]), keep=1) == [0, 1, 2]
    with pytest.raises(ValueError):
        shortlist(torch.tensor([[0.0], [float("nan")]]))
    with pytest.raises(ValueError):
        shortlist(torch.zeros(2, 1), keep=-1)


def test_screen_uses_five_step_action_packets(monkeypatch):
    seen = []
    def successor(model, pixels, history, prefix, **kwargs):
        return prefix, prefix
    def terminal(model, embeddings, actions, probes, **kwargs):
        seen.append(probes.clone())
        return probes[:, 0, :] * (1 + embeddings[:, 0, :])
    monkeypatch.setattr(module.p1, "_state_after_prefix", successor)
    monkeypatch.setattr(module.p1, "_terminal_from_state", terminal)
    monkeypatch.setattr(module.p0, "_planning_coordinate", lambda model, x, **kwargs: x)
    bank = torch.stack([torch.zeros(1, 10), torch.ones(1, 10), torch.ones(1, 10) * 2])
    result, receipt = module.screen(None, None, None, bank, lower=None, upper=None)
    assert result.shape == (3, 1, 10)
    assert seen[0].shape == (5, 1, 10)
    assert torch.allclose(seen[0][1, 0], torch.tensor([0.3, 0.] * 5))
    assert receipt["suffix_predict_samples"] == 15
