from copy import deepcopy

import pytest

from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatcher
from experiments.spectral_reachability.audit_reacher_d2p_program_dispatch_result import summarize
from test_reacher_d2p_program_dispatch import _receipt
from test_reacher_d2p_program_dispatch_diagnostic import _methods


def _report():
    rows = []
    for seed, witness in ((1, False), (2, True)):
        methods = _methods()
        receipt = _receipt(witness)
        decision = dispatcher.dispatch_decision(receipt)
        methods["dispatch"]["first_receipt"] = {"dispatch": decision,
            "reconfiguration_compilation": receipt,
            "model_predict_sample_evaluations": 48, "backward_vjp_rows": 576}
        first_samples = {"dispatch": 48, "same_bank_greedy": 3, "zero": 0, "lookahead_cem": 12}
        for name, method in methods.items():
            method["suffix_receipt"] = {"model_predict_sample_evaluations": 14, "backward_vjp_rows": 2}
            method["actual_compute"] = {"model_predict_sample_evaluations": first_samples[name] + 14,
                "model_predict_calls": 10, "logical_backward_vjp_rows": method["first_receipt"].get("backward_vjp_rows", 0) + 2}
            method["planner_wall_seconds"] = 1.
            method["physical"]["after_first_chunk_qpos_distance"] = 1.
            method["observed_first_chunk"] = {"candidate_goal_cost": 1.}
        methods["zero"]["physical"]["final_qpos_distance"] = 1.5
        if witness:
            methods["dispatch"]["action_witness_raw"] = [[0., 0.]]
            methods["dispatch"]["physical"]["final_qpos_distance"] = .2
        rows.append({"seed": seed, "source_episode": seed + 10, "methods": methods})
    return {"schema": "reacher_d2p_program_dispatch_diagnostic.v1", "status": "completed", "gates": {"intact": True},
        "sealed_protocol": {"root_seeds": [1, 2], "proposals": 3, "policy": dispatcher.POLICY,
            "cem": {"samples": 2, "steps": 2, "horizon": 3}}, "rows": rows}


def test_full_root_effect_and_work_reconciliation():
    result = summarize(_report())
    assert result["reconfiguration_count"] == result["progress_fallback_count"] == 1
    assert result["paired"]["dispatch_minus_greedy"]["mean"] == pytest.approx(-.15)
    assert result["paired"]["dispatch_minus_neutral"]["mean"] == pytest.approx(-.5)
    assert result["compute"]["dispatch"]["logical_backward_vjp_rows"]["mean"] == 578


def test_result_is_root_order_invariant():
    report = _report()
    expected = summarize(report)
    report["rows"].reverse()
    assert summarize(report) == expected


@pytest.mark.parametrize("kind", ["root", "work", "backward", "routing", "gate"])
def test_fail_closed_on_corrupt_diagnostic(kind):
    report = deepcopy(_report())
    method = report["rows"][0]["methods"]["dispatch"]
    if kind == "root":
        report["rows"].pop()
    elif kind == "work":
        method["actual_compute"]["model_predict_sample_evaluations"] -= 1
    elif kind == "backward":
        method["actual_compute"]["logical_backward_vjp_rows"] = 0
    elif kind == "routing":
        method["first_receipt"]["dispatch"]["selected_program_index"] = 0
    else:
        report["gates"]["intact"] = False
    with pytest.raises(ValueError):
        summarize(report)
