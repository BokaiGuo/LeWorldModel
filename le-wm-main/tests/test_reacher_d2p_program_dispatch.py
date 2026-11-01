from copy import deepcopy

import pytest
import torch

from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatch


def _receipt(witness=False):
    return {"future_reference": "same_bank_greedy", "recompiled_suffix_solver": dispatch.compiler.SUFFIX_SOLVER,
        "proposal_rows": [{"proposal_index": 0, "immediate_goal_cost": 2., "bounded_dls_suffix_goal_cost": 3.},
            {"proposal_index": 1, "immediate_goal_cost": .5, "bounded_dls_suffix_goal_cost": 1.},
            {"proposal_index": 2, "immediate_goal_cost": 2.1, "bounded_dls_suffix_goal_cost": .3}],
        "future_reference_proposal_index": 1, "non_greedy_witness_found": witness,
        "selected_proposal_index": 2 if witness else None, "selection_metric": "bounded_dls_suffix_goal_cost",
        "future_reference_score_improvement": .7,
        "forward_rollout_evaluations": 27, "model_predict_sample_evaluations": 48, "backward_vjp_rows": 576}


def test_no_witness_dispatches_goal_progress_not_neutral_control():
    result = dispatch.dispatch_decision(_receipt())
    assert result["route"] == "same_bank_goal_progress"
    assert result["selected_program_index"] == 1
    assert not result["reconfiguration_witness_found"]


def test_physical_labels_cannot_change_dispatch_or_model_eligibility():
    receipt = _receipt(True)
    result = dispatch.dispatch_decision(receipt)
    receipt["physical"] = {"greedy_final_distance": 0, "selected_final_distance": 100}
    assert dispatch.dispatch_decision(receipt) == result
    assert result["selected_program_index"] == 2


@pytest.mark.parametrize("field,value", [
    ("future_reference", "zero_prefix"), ("recompiled_suffix_solver", "old"),
    ("future_reference_proposal_index", 0), ("selected_proposal_index", 99),
    ("future_reference_score_improvement", -1.),
    ("selection_metric", "unknown"),
])
def test_dispatch_rejects_stale_or_malformed_witness(field, value):
    receipt = _receipt(True)
    receipt[field] = value
    with pytest.raises(ValueError):
        dispatch.dispatch_decision(receipt)


@pytest.mark.parametrize("witness", [False, True])
def test_compilation_returns_exact_action_and_preserves_measured_costs(monkeypatch, witness):
    receipt = _receipt(witness)
    bank = torch.arange(3.).reshape(3, 1, 1)
    monkeypatch.setattr(dispatch.compiler, "_compile_reconfiguration", lambda *args, **kwargs:
        (bank[2:3] if witness else None, deepcopy(receipt)))
    params, compiled = dispatch.compile_program_dispatch(object(), None, None, torch.zeros(1, 3, 1), proposal_bank=bank)
    assert torch.equal(params, bank[2:3] if witness else bank[1:2])
    assert compiled["backward_vjp_rows"] == receipt["backward_vjp_rows"]
