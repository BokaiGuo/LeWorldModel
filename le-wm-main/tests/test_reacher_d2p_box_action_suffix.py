"""Algebra fixtures, not CPU neural efficacy experiments."""
import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_box_action_suffix as box
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


class AffineFrozen:
    def action_encoder(self, value):
        return value

    def predict(self, emb, actions):
        return actions[..., :1]


def inputs():
    return dict(model=AffineFrozen(), goal_latent=torch.tensor([.25]), emb_context=torch.zeros(1, 3, 1),
        action_context=torch.zeros(1, 3, 1), initial_commands=torch.ones(1, 2, 1),
        action_lower=torch.tensor([-1.]), action_upper=torch.tensor([1.]))


def test_command_and_legacy_forward_exact_including_saturated_boundary():
    kw = inputs()
    for magnitude in (.2, 20.):
        params = torch.full((1, 2, 1), magnitude)
        baseline = p1._terminal_from_state(kw["model"], kw["emb_context"], kw["action_context"], params,
            action_lower=kw["action_lower"], action_upper=kw["action_upper"])
        actual = box.terminal_commands(kw["model"], kw["emb_context"], kw["action_context"], params.tanh(),
            action_lower=kw["action_lower"], action_upper=kw["action_upper"])
        assert torch.equal(actual, baseline)


def test_box_can_leave_exact_tanh_saturation_without_input_mutation():
    kw = inputs()
    saved = kw["initial_commands"].clone()
    results = box.compile_box_snapshots(**kw)
    commands, receipt = results[8]
    assert torch.equal(kw["initial_commands"], saved)
    assert commands.abs().max() <= 1
    assert abs(float(commands[0, -1, 0]) - .25) < 1e-3
    assert not commands.requires_grad
    assert receipt["cost_nonincreasing"]
    assert receipt["trace"][-1]["cost_after"] < receipt["trace"][0]["cost_before"]
    assert receipt["model_predict_sample_evaluations"] == receipt["terminal_forward_evaluations"] * 2


@pytest.mark.parametrize("value", [None, [[[0.]]], torch.zeros(1, 2, 1, dtype=torch.float64),
    torch.full((1, 2, 1), float("nan")), torch.full((1, 2, 1), 1.1)])
def test_initial_command_validation(value):
    kw = inputs()
    kw["initial_commands"] = value
    with pytest.raises(ValueError):
        box.compile_box_snapshots(**kw)


@pytest.mark.parametrize("checkpoints", [(), (1, 1), (4, 2), (0,), (True,)])
def test_invalid_checkpoints(checkpoints):
    with pytest.raises(ValueError):
        box.compile_box_snapshots(**inputs(), checkpoints=checkpoints)
