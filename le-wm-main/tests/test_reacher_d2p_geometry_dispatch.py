import copy

import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_geometry_dispatch as module
from experiments.spectral_reachability.audit_reacher_d2p_post_prefix_geometry import geometry_metrics


def receipt():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1},
        {"proposal_index": 3, "immediate_goal_cost": 1.4, "refined_goal_cost": .2}]
    return {"proposal_rows": rows, "non_greedy_margin": 1e-6,
        "dispatch": {"same_bank_greedy_index": 1, "selected_program_index": 2}}


def geometry(gains):
    return {index: {"bounded_quadratic_gain": gain} for index, gain in enumerate(gains)}


def test_endpoint_best_without_geometry_not_reconfiguration():
    decision = module.qualify_decision(receipt(), geometry([.3, .4, .2, .5]))
    assert decision["selected_program_index"] == 3
    assert decision["geometry_qualified_indices"] == [3]
    assert decision["physical_labels_used"] is False


def test_both_controls_must_be_beaten_not_just_greedy():
    decision = module.qualify_decision(receipt(), geometry([.6, .4, .5, .6]))
    assert decision["selected_program_index"] == 1
    assert not decision["reconfiguration_witness_found"]


def test_missing_geometry_for_any_endpoint_winner_rejected():
    with pytest.raises(ValueError, match="every"):
        module.qualify_decision(receipt(), {0: {"bounded_quadratic_gain": 0}})


def test_nonfinite_geometry_rejected():
    with pytest.raises(ValueError, match="finite"):
        module.qualify_decision(receipt(), geometry([.3, .4, float("nan"), .5]))


def test_tie_uses_original_refined_cost_then_index():
    value = receipt()
    assert module.qualify_decision(value, geometry([.3, .4, .5, .5]))["selected_program_index"] == 2
    value["proposal_rows"][3]["refined_goal_cost"] = .1
    assert module.qualify_decision(value, geometry([.3, .4, .5, .5]))["selected_program_index"] == 2


def test_near_goal_own_residual_cannot_replace_common_residual():
    response = torch.eye(2, dtype=torch.float64)
    common = torch.tensor([1., 1.], dtype=torch.float64)
    stronger = module.bounded_response_gain(response * 2, common)
    weaker = module.bounded_response_gain(response, common)
    assert stronger["bounded_quadratic_gain"] > weaker["bounded_quadratic_gain"]
    assert module.bounded_response_gain(response, torch.zeros_like(common))["bounded_quadratic_gain"] == 0


def test_generic_gram_solver_matches_physical_diagnostic():
    generator = torch.Generator().manual_seed(82)
    response = torch.randn(2, 20, generator=generator, dtype=torch.float64) * .1
    residual = torch.tensor([.8, -.2], dtype=torch.float64)
    expected = geometry_metrics(response, residual)
    actual = module.bounded_response_gain(response, residual)
    assert actual["bounded_quadratic_gain"] == pytest.approx(expected["bounded_quadratic_gain"], abs=1e-12)


def test_compile_predict_and_vjp_accounting(monkeypatch):
    value = receipt()
    value.update({"model_predict_sample_evaluations": 100, "backward_vjp_rows": 200})
    monkeypatch.setattr(module.base, "compile_refined_dispatch", lambda *a, **kw: (None, copy.deepcopy(value)))
    monkeypatch.setattr(module.p0, "_goal_latent", lambda *a: torch.ones(2))
    monkeypatch.setattr(module.p0, "_planning_coordinate", lambda _model, x, **kw: x)
    monkeypatch.setattr(module.p1, "_state_after_prefix", lambda _m, _p, _h, prefix, **kw: (prefix.expand(1, 2, 2), torch.zeros(1, 2, 2)))
    def terminal(_m, emb, _act, future, **kw):
        return future.sum(1) * (emb[0, 0, 0] + 1)
    monkeypatch.setattr(module.p1, "_terminal_from_state", terminal)
    bank = torch.arange(4.).reshape(4, 1, 1).expand(4, 1, 2)
    _, result = module.compile_refined_dispatch(object(), torch.zeros(1), torch.zeros(1), torch.zeros(1, 2, 2),
        proposal_bank=bank, action_lower=torch.tensor([-1., -1.]), action_upper=torch.tensor([1., 1.]))
    assert result["model_predict_sample_evaluations"] == 120
    assert result["backward_vjp_rows"] == 208
    assert set(result["common_task_response"]["candidate_metrics"]) == {0, 1, 2, 3}
