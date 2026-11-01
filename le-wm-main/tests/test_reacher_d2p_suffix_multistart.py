"""Algebra/protocol fixtures; no CPU neural efficacy."""
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_suffix_multistart as module
from experiments.spectral_reachability.reacher_d2p_suffix_multistart import NAMES, choose_candidate


def test_incumbent_wins_ties_without_outcome_labels():
    assert choose_candidate(dict.fromkeys(NAMES, 1.)) == "incumbent"


def test_restart_can_escape_high_local_model_cost():
    costs = dict.fromkeys(NAMES, 1.6)
    costs["incumbent"] = 1.8
    costs["same_prefix_cem"] = .006
    assert choose_candidate(costs) == "same_prefix_cem"


def test_box_has_no_unconditional_preference():
    costs = dict.fromkeys(NAMES, .02)
    costs["incumbent"] = .005
    assert choose_candidate(costs) == "incumbent"
    costs["box_extra_8"] = .003
    assert choose_candidate(costs) == "box_extra_8"


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -.1])
def test_finite_nonnegative_costs_required(value):
    costs = dict.fromkeys(NAMES, 1.)
    costs["same_prefix_cem"] = value
    with pytest.raises(ValueError):
        choose_candidate(costs)


def test_no_missing_candidate():
    with pytest.raises(ValueError):
        choose_candidate({"incumbent": 1.})


def test_compile_candidate_accounting_and_command_output(monkeypatch):
    class Affine:
        samples = 0

        def action_encoder(self, value):
            return value

        def predict(self, emb, actions):
            self.samples += emb.shape[0]
            return actions[..., :1]

    model = Affine()
    emb, history = torch.zeros(1, 3, 1), torch.zeros(1, 3, 1)
    goal = torch.tensor([.25])
    lower, upper = torch.tensor([-1.]), torch.tensor([1.])
    monkeypatch.setattr(module.p0, "_latent_context", lambda *a: emb)
    monkeypatch.setattr(module.p0, "_goal_latent", lambda *a: goal)

    def cost_fn(model, pixels, goal_image, history, params, **bounds):
        latent = module.p1._terminal_from_state(model, emb.expand(len(params), -1, -1),
            history.expand(len(params), -1, -1), params, **bounds)
        return (latent - goal).square().mean(dim=1)

    monkeypatch.setattr(module.p0, "_cost_fn", cost_fn)
    initial = torch.full((1, 2, 1), .2)
    saved = initial.clone()
    commands, receipt = module.compile_candidates(model, None, None, history, incumbent=initial,
        action_lower=lower, action_upper=upper, seed=71)
    assert set(commands) == set(NAMES)
    assert torch.equal(initial, saved)
    assert all(value.shape == initial.shape and value.abs().max() <= 1 for value in commands.values())
    assert receipt["model_predict_sample_evaluations"] == model.samples
    assert receipt["model_cost_not_above_incumbent"]
    assert receipt["physical_labels_used"] is False
