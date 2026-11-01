"""Analytic tensor fixtures, not neural modules, model loading or GPU."""
import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_h4_parfit_20261101 as h4
from experiments.spectral_reachability import p1_node_h4_parfit_20261101 as node


def test_feedback_mix_keeps_four_response_slots_and_seeded_gaussian():
    bank = np.arange(8, dtype=np.float32)[:, None, None, None] * np.ones((8, 4, 5, 2), np.float32) / 8
    np.testing.assert_array_equal(node.feedback_mix(bank, 0), bank)
    mixed = node.feedback_mix(bank, 5)
    np.testing.assert_array_equal(mixed[:4], bank[[3, 5, 6, 7]])
    np.testing.assert_array_equal(mixed, node.feedback_mix(bank, 5))
    assert not np.array_equal(mixed[4:], node.feedback_mix(bank, 10)[4:])


class TensorFixture:
    """Counter-only leaf implementing an elementary differentiable expression."""
    def __init__(self, caps):
        self.driver = type("Counter", (), {})()
        self.driver.caps = caps.copy()
        self.driver.charged = dict.fromkeys(h4.AXES, 0)
        self.driver.failed = self.driver.closed = False

    def _dispatch(self, phase, kind, cost, fn):
        for key, amount in cost.items():
            if self.driver.charged[key] + amount > self.driver.caps[key]:
                self.driver.failed = True
                raise ValueError("budget exceeded")
        for key, amount in cost.items():
            self.driver.charged[key] += amount
        try:
            return fn()
        except BaseException:
            self.driver.failed = True
            raise

    def action_encode(self, actions, *, phase):
        return self._dispatch(phase, "action_encode", {"action_encode_calls": 1}, lambda: actions)

    def predict(self, states, actions, *, phase):
        def evaluate():
            scalar = actions[:, -1, :].mean(dim=-1, keepdim=True)
            return states + scalar[:, None, :]
        return self._dispatch(phase, "predict", {"predict_calls": 1, "predict_samples": 1}, evaluate)

    def gradient(self, loss, actions, *, phase):
        return self._dispatch(phase, "backward", {"backward_calls": 1, "vjp_rows": 1},
                              lambda: torch.autograd.grad(loss, actions)[0])

    def _abort(self):
        self.driver.failed = True


def inputs(boundary):
    h, carry = h4.dimensions(boundary)
    prior = torch.zeros(h + 1, 5, 2) if carry else None
    return dict(boundary=boundary, states=torch.zeros(1, 3, 192),
                history=torch.zeros(1, 2, 10), goal=torch.ones(1, 192),
                mean=torch.zeros(2), std=torch.ones(2), matrix=torch.eye(192),
                previous=prior, issued=prior[0].clone() if carry else None)


@pytest.mark.parametrize("arm", h4.ARMS)
@pytest.mark.parametrize("boundary", [0, 5, 10, 15])
def test_full_tensor_node_charges_and_selection(arm, boundary):
    leaf = TensorFixture(h4.node_cost(arm, boundary))
    events = []
    result = node.run_node(leaf, arm=arm, **inputs(boundary),
                           emit=lambda kind, payload: events.append((kind, payload)))
    h, carry = h4.dimensions(boundary)
    assert result["selected_raw"].shape == (h, 5, 2)
    assert result["optimized_raw"].shape == (8, 4, 5, 2)
    assert result["charges"] == leaf.driver.charged == h4.node_cost(arm, boundary)
    assert len(result["scores"]) == (9 if carry else 8)
    assert len([1 for name, _ in events if name == "suffix_exchange"]) == (h > 1)
    assert not leaf.driver.failed


def test_preflight_and_action_mismatch_do_not_dispatch():
    data = inputs(5)
    leaf = TensorFixture(h4.node_cost("random", 5))
    data["issued"] = torch.ones(5, 2)
    with pytest.raises(ValueError, match="packet mismatch"):
        node.run_node(leaf, arm="random", **data, emit=lambda *_: None)
    assert not any(leaf.driver.charged.values())
    leaf.driver.caps["predict_calls"] -= 1
    with pytest.raises(ValueError, match="reservation"):
        node.run_node(leaf, arm="random", **inputs(5), emit=lambda *_: None)
    assert not any(leaf.driver.charged.values())


def test_failed_prediction_is_charged_and_node_stops():
    leaf = TensorFixture(h4.node_cost("random", 0))
    def fail(*args, **kwargs):
        return leaf._dispatch("common_suffix", "predict",
                              {"predict_calls": 1, "predict_samples": 1},
                              lambda: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    leaf.predict = fail
    with pytest.raises(RuntimeError, match="fixture failure"):
        node.run_node(leaf, arm="random", **inputs(0), emit=lambda *_: None)
    assert leaf.driver.failed and leaf.driver.charged["predict_calls"] == 1
