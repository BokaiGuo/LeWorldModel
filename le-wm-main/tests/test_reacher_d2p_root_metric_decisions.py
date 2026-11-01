"""Numeric report fixtures only, no neural inference or physical efficacy."""
import copy

import pytest

from experiments.spectral_reachability import audit_reacher_d2p_root_metric_decisions as audit
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler


def reports():
    bank = [{"index":i,"prefix_cost":1. if i==0 else 2.,"terminal_cost":2. if i==0 else 1.,
        "common_response_gain":{"bounded_quadratic_gain":1. if i==0 else 2.,
            "bounded_quadratic_gain_upper_bound":1. if i==0 else 2.,
            "objective_certificate_passed":True,"physical_labels_used":False}} for i in range(17)]
    decision = compiler.select(bank)
    programs = [{"index":i,"fits":{"model":{"gain":1. if i==0 else 2.}},
        "original_physical_diagnostic_conjunction":i==2,
        "physical_rgb_positive_point_margin_both_scales":i==2,
        "physical_rgb_gain_minus_fixed_controls":[1.,1.] if i==2 else [-1.,-1.]} for i in range(17)]
    score = {"status":"completed","gates":{"cuda":True},"actual_certified_qps":408,"rows":[]}
    diagnostic = {"independent_metric_fits":1224,"rows":[]}
    for seed in range(8):
        score["rows"].append({"seed":seed,"tasks":{task:{"bank":copy.deepcopy(bank),"decision":decision,
            "original_decision":decision} for task in audit.TASKS}})
        diagnostic["rows"].append({"seed":seed,"tasks":{task:{"fixed_greedy_index":0,
            "programs":copy.deepcopy(programs)} for task in audit.TASKS}})
    return score, diagnostic


def test_actual_selected_failure_is_not_replaced_by_physical_best():
    a,b = reports()
    before = copy.deepcopy((a,b))
    result = audit.evaluate(a,b)
    assert result["independent_certificates_checked"]==408
    assert result["aggregate"]["shoulder_only"]["model_witness_roots"]==8
    assert result["aggregate"]["shoulder_only"]["selected_passes_both_diagnostic_checks"]==0
    assert all(r["tasks"]["shoulder_only"]["selected_index"]==1 for r in result["rows"])
    assert not result["P1_admission"]
    assert (a,b)==before


def test_root_identity_join_ignores_input_order():
    a,b = reports()
    expected = audit.evaluate(a,b)
    b["rows"].reverse()
    assert audit.evaluate(a,b)==expected


@pytest.mark.parametrize("bad",["duplicate","missing","certificate","selector","program_order","failed_gate"])
def test_invalid_evidence_fails_closed(bad):
    a,b = reports()
    if bad=="duplicate": b["rows"][0]["seed"]=1
    elif bad=="missing": b["rows"].pop()
    elif bad=="certificate": b["rows"][0]["tasks"]["full_drive"]["programs"][0]["fits"]["model"]["gain"]=10.
    elif bad=="selector": a["rows"][0]["tasks"]["full_drive"]["decision"]={"native_index":2}
    elif bad=="program_order": b["rows"][0]["tasks"]["full_drive"]["programs"].reverse()
    else: a["gates"]["cuda"]=False
    with pytest.raises(ValueError): audit.evaluate(a,b)
