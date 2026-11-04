import pytest
import torch

from experiments.spectral_reachability import reacher_pc_prefix_refinement as module

refine = module.refine


def test_refinement_changes_seed_but_respects_trust_region():
    initial = torch.tensor([0.1, 0.0, 0.0])
    def evaluate(x):
        return x.square().sum(), torch.stack([(x - 1).square().sum(), (x + 1).square().sum()])
    output, trace = refine(initial, evaluate, neutral_cost=torch.tensor(0.0))
    assert 0 < float((output - initial).norm()) <= 0.250001
    assert len(trace) == 4
    assert all(len(row["future_costs"]) == 2 for row in trace)


def test_rejects_nonfinite_and_invalid_configuration():
    with pytest.raises(ValueError):
        refine(torch.zeros(2), lambda x: (x.sum(), x * float("nan")), torch.tensor(0.0))
    with pytest.raises(ValueError):
        refine(torch.zeros(2), lambda x: (x.sum(), x), torch.tensor(0.0), radius=0)


def test_augment_preserves_controls_and_accounts_for_calls(monkeypatch):
    monkeypatch.setattr(module.p0, "_goal_latent", lambda model, goal: torch.zeros(1, 10))
    monkeypatch.setattr(module.p0, "_planning_coordinate", lambda model, x, **kwargs: x)
    monkeypatch.setattr(module.p1, "_state_after_prefix",
                        lambda model, pixels, history, prefix, **kwargs: (prefix, prefix))
    monkeypatch.setattr(module.p1, "_terminal_from_state",
                        lambda model, emb, actions, probes, **kwargs: emb[:, 0, :] + probes[:, 0, :])
    bank = torch.stack([torch.zeros(1, 10), torch.ones(1, 10),
                        torch.ones(1, 10) * 0.2])
    result, receipt = module.augment(None, None, None, None, bank, lower=None, upper=None)
    assert result.shape == (4, 1, 10)
    assert torch.equal(result[:3], bank)
    assert receipt["refinement_forward_prefix_samples"] == 5
    assert receipt["refinement_forward_suffix_samples"] == 25
    assert receipt["refinement_backward_calls"] == 4
