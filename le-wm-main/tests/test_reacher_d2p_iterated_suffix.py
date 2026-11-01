import pytest
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.reacher_d2p_iterated_suffix import compile_snapshots


def inputs(monkeypatch):
    monkeypatch.setattr(p1, "_terminal_from_state", lambda _model, _emb, _history, params, **kw: params[:, 0].tanh())
    return dict(model=object(), goal_latent=torch.tensor([.5]), emb_context=torch.zeros(1, 3, 1),
        action_context=torch.zeros(1, 3, 1), future_horizon=1, action_lower=torch.tensor([-1.]),
        action_upper=torch.tensor([1.]), damping=.1)


def test_first_step_is_exact_legacy_svd_solver(monkeypatch):
    kw = inputs(monkeypatch)
    old = p1._reachability_value(**kw, return_suffix_params=True)
    snapshots = compile_snapshots(**kw)
    params, receipt = snapshots[1]
    assert torch.equal(params, old["_bounded_dls_suffix_params"])
    assert receipt["trace"][0]["cost_after"] == old["bounded_dls_suffix_goal_cost"]
    assert receipt["model_predict_sample_evaluations"] == old["model_predict_sample_evaluations"]
    assert receipt["backward_vjp_rows"] == old["backward_vjp_rows"]


def test_relinearization_improves_frozen_model_objective(monkeypatch):
    snapshots = compile_snapshots(**inputs(monkeypatch))
    assert tuple(snapshots) == (1, 2, 4, 8)
    costs = [receipt["trace"][-1]["cost_after"] for _, receipt in snapshots.values()]
    assert costs == sorted(costs, reverse=True)
    assert costs[-1] < costs[0]
    assert all(receipt["cost_nonincreasing"] and not receipt["physical_labels_used"] for _, receipt in snapshots.values())
    assert all(not params.requires_grad for params, _ in snapshots.values())


def test_explicit_zero_start_preserves_default_actions_and_trace(monkeypatch):
    kw = inputs(monkeypatch)
    baseline = compile_snapshots(**kw)
    initialized = compile_snapshots(**kw, initial_params=torch.zeros(1, 1, 1))
    for step, (params, receipt) in baseline.items():
        other_params, other = initialized[step]
        assert torch.equal(params, other_params)
        assert receipt == {key: value for key, value in other.items() if key != "initialization"}


def test_warm_start_preserves_input_and_never_increases_initial_objective(monkeypatch):
    kw = inputs(monkeypatch)
    initial = torch.tensor([[[.8]]], requires_grad=True)
    saved = initial.detach().clone()
    params, receipt = compile_snapshots(**kw, initial_params=initial)[8]
    assert torch.equal(initial.detach(), saved)
    assert not params.requires_grad
    assert receipt["initialization"] == "explicit_planned_suffix"
    assert receipt["trace"][0]["cost_before"] == float((saved.tanh() - .5).square().mean())
    assert receipt["trace"][-1]["cost_after"] <= receipt["trace"][0]["cost_before"]


@pytest.mark.parametrize("initial", [torch.zeros(1), torch.zeros(1, 2, 1),
    torch.zeros(1, 1, 1, dtype=torch.float64), torch.full((1, 1, 1), float("nan")), [[[0.]]]])
def test_warm_start_rejects_silent_shape_dtype_or_finiteness_coercion(monkeypatch, initial):
    with pytest.raises(ValueError, match="finite initial parameters"):
        compile_snapshots(**inputs(monkeypatch), initial_params=initial)


@pytest.mark.parametrize("checkpoints", [(), (0,), (2, 1), (1, 1), (1.5,)])
def test_rejects_invalid_snapshots(monkeypatch, checkpoints):
    with pytest.raises(ValueError):
        compile_snapshots(**inputs(monkeypatch), checkpoints=checkpoints)


@pytest.mark.parametrize("damping", [0., -1., float("nan"), float("inf")])
def test_rejects_invalid_damping(monkeypatch, damping):
    kw = inputs(monkeypatch)
    kw["damping"] = damping
    with pytest.raises(ValueError):
        compile_snapshots(**kw)
