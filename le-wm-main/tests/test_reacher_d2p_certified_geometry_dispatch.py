import torch

from experiments.spectral_reachability import reacher_d2p_certified_geometry_dispatch as compiler


def test_certified_compiler_binds_geometry_and_accounts_extra_work(monkeypatch):
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1}]
    original = compiler.base.select_refined_bank(rows)
    receipt = {"proposal_rows": rows, "non_greedy_margin": 1e-6, "dispatch": original,
        "model_predict_sample_evaluations": 100, "backward_vjp_rows": 200}
    monkeypatch.setattr(compiler.base, "compile_refined_dispatch", lambda *a, **kw: (None, receipt))
    metrics = {index: {"bounded_quadratic_gain": gain, "bounded_quadratic_gain_upper_bound": gain + 1e-12,
        "objective_certificate_passed": True} for index, gain in enumerate([.3, .4, .5])}
    def response(*args, **kwargs):
        assert kwargs["required"] == [0, 1, 2]
        assert kwargs["gain_solver"] is compiler.qp.certified_response_gain
        return metrics, torch.ones(2), 15, 6
    monkeypatch.setattr(compiler.geometry, "response_bank_metrics", response)
    bank = torch.arange(3.).reshape(3, 1, 1)
    chosen, result = compiler.compile_refined_dispatch(object(), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        proposal_bank=bank, action_lower=torch.tensor([-1.]), action_upper=torch.tensor([1.]))
    assert torch.equal(chosen, bank[2:3])
    assert result["dispatch"]["objective_comparison_certified"]
    assert result["model_predict_sample_evaluations"] == 115
    assert result["backward_vjp_rows"] == 206
    assert result["endpoint_only_dispatch"] == original
