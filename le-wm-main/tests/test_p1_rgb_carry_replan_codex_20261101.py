import pytest
import torch

from experiments.spectral_reachability import p1_rgb_carry_replan_codex_20261101 as m


class Leaf:
    def __init__(self):
        self.calls = {}

    def _dispatch(self, phase, operation, cost, fn):
        for key, value in cost.items():
            self.calls[key] = self.calls.get(key, 0) + value
        return fn()

    def action_encode(self, context, *, phase):
        self.calls["action_encode_calls"] = self.calls.get("action_encode_calls", 0) + 1
        return context

    def predict(self, states, actions, *, phase):
        self.calls["predict_calls"] = self.calls.get("predict_calls", 0) + 1
        self.calls["predict_samples"] = self.calls.get("predict_samples", 0) + len(states)
        result = states.clone()
        result[:, -1, :] += actions[:, -1, :].mean(dim=-1, keepdim=True)
        return result


def inputs():
    return dict(states=torch.zeros(1, 3, 2), history=torch.zeros(1, 2, 10),
                goal=torch.ones(1, 2), mean=torch.zeros(2), std=torch.ones(2),
                new_candidates=torch.zeros(2, 3, 5, 2))


def test_carry_wins_and_cost_exact():
    args = inputs()
    previous = torch.zeros(3, 5, 2)
    previous[1:] = .5
    leaf = Leaf()
    result = m.select(leaf, **args, previous_selected=previous)
    assert result["selected_source"] == "carry"
    assert torch.equal(result["selected_raw"], previous[1:])
    assert result["horizon_packets"] == 2
    for key, value in leaf.calls.items():
        assert result["incremental_cost"][key] == value
    assert result["incremental_cost"]["predict_calls"] == 6


def test_current_context_changes_selection_and_ties_keep_carry():
    args = inputs()
    previous = torch.zeros(3, 5, 2)
    previous[1:] = .5
    assert m.select(Leaf(), **args, previous_selected=previous)["selected_source"] == "carry"
    args["states"][:, -1] = 1
    assert m.select(Leaf(), **args, previous_selected=previous)["selected_source"] == "new"
    args["new_candidates"][:] = .5
    assert m.select(Leaf(), **args, previous_selected=previous)["selected_source"] == "carry"


def test_invalid_inputs_fail_closed():
    args = inputs()
    with pytest.raises(ValueError):
        m.select(Leaf(), **args, previous_selected=torch.zeros(1, 5, 2))
    args["new_candidates"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        m.select(Leaf(), **args)
