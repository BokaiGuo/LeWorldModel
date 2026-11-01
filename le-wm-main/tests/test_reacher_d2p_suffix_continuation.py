import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_suffix_continuation as continuation
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import choose_model_incumbent


@pytest.mark.parametrize("cold,warm", [(0., 0.), (1., 0.), (0., 1.)])
def test_runtime_and_sealed_diagnostic_share_selection_rule(cold, warm):
    costs = {"cold": cold, "warm": warm}
    assert continuation.choose_incumbent(costs) == choose_model_incumbent(costs)


def test_runtime_uses_same_solver_and_preserves_cold_fallback(monkeypatch):
    calls = []

    def mock_solver(*args, **kw):
        calls.append(kw)
        warm = kw.get("initial_params") is not None
        value = 2. if warm else 1.
        receipt = {"trace": [{"cost_before": 3., "cost_after": value}],
            "model_predict_sample_evaluations": 10, "backward_vjp_rows": 20}
        return {8: (torch.full((1, 2, 1), value), receipt)}

    monkeypatch.setattr(continuation.solver, "compile_snapshots", mock_solver)
    plan = torch.zeros(1, 2, 1)
    variants, receipt = continuation.compile_continuation(object(), None, None, None,
        planned_suffix=plan, action_lower=-1., action_upper=1.)
    assert calls[0]["initial_params"] is plan
    assert "initial_params" not in calls[1]
    assert receipt["selected"] == "cold"
    assert receipt["model_predict_sample_evaluations"] == 20
    assert receipt["backward_vjp_rows"] == 40
    assert receipt["physical_labels_used"] is False
    assert variants["carry"] is not plan
    assert all(call["checkpoints"] == (8,) for call in calls)
