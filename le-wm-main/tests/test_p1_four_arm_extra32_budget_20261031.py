from experiments.spectral_reachability import p1_four_arm_extra32_budget_20261031 as extra
from experiments.spectral_reachability import p1_four_arm_extra32_root1_20261031 as run
from experiments.spectral_reachability import p1_four_arm_winner_budget_20261031 as previous
import pytest
import torch


def test_only_selected_baselines_receive_fixed_extra_iterations():
    for arm in ("structured", "immediate_progress", "native", "random"):
        old, new = previous.cost(arm), extra.cost(arm)
        if arm in extra.EXTRA_ARMS:
            # Prior contract already included two extra updates; new one has 32.
            assert new["backward_calls"] - old["backward_calls"] == 30
            assert new["vjp_rows"] - old["vjp_rows"] == 30
            assert new["predict_calls"] - old["predict_calls"] == 90
            assert new["action_encode_calls"] - old["action_encode_calls"] == 90
            assert new["replay_calls"] == old["replay_calls"]
            assert new["selection_evaluations"] == old["selection_evaluations"]
        else:
            assert new == old


def test_contract_is_fixed_before_outcomes():
    assert extra.EXTRA_ROUNDS == 16
    assert extra.EXTRA_ARMS == ("native", "random")
    assert extra.base.CONFIG["steps"] == 2


@pytest.mark.parametrize("declared", [1, 2])
def test_autograd_meter_detects_mismatch_and_restores_api(monkeypatch, declared):
    class FakeHooks:
        counts = {"predict": 0, "action_encode": 0}
        def __init__(self, model):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class Store:
        def put(self, name, value):
            self.record = value

    def fake_plan(*args):
        x = torch.tensor(2.0, requires_grad=True)
        torch.autograd.grad(x * x, x)
        return {"account": {"charged_upper_bounds": {
            "backward_calls": declared, "predict_calls": 0, "action_encode_calls": 0}}}

    monkeypatch.setattr(run.hooks, "ModuleCalls", FakeHooks)
    monkeypatch.setattr(run.bridge, "plan_node", fake_plan)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    original = torch.autograd.grad
    store = Store()
    runtime = {"device": torch.device("cuda:0"), "model": None}
    if declared == 1:
        result = run.plan_node({}, "native", 0, store, runtime)
        assert result["module_hook_audit"]["autograd_api_counts"]["grad_completed"] == 1
    else:
        with pytest.raises(ValueError, match="autograd API/ledger mismatch"):
            run.plan_node({}, "native", 0, store, runtime)
    assert torch.autograd.grad is original
