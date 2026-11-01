from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_variable_cem_paired_cost_audit_20261101_leibniz as audit
from test_p1_full_cem_cost_compare_20261101_main import fixture


def pair():
    structured, cem = fixture("structured"), fixture("cem")
    for report in (structured, cem):
        nodes = next(iter(report["nodes"].values()))
        nodes.extend([{"boundary": 10, "status": "not_started", "receipt": None},
                      {"boundary": 15, "status": "not_started", "receipt": None}])
    return structured, cem


def test_missing_nodes_and_unmetered_selection_not_zero():
    result = audit.paired(*pair(), 0)
    assert result["arms"][0]["nodes"][1]["charge"] is None
    assert result["arms"][1]["nodes"][0]["charge"]["selection_evaluations"] is None
    assert result["arms"][1]["nodes"][0]["charge"]["predict_calls"] == 2
    assert result["equal_total_compute"] is False


@pytest.mark.parametrize("defect", ["goal", "model", "roster", "cap", "hook"])
def test_rejects_mismatch(defect):
    structured, cem = deepcopy(pair())
    if defect == "goal":
        cem["cp10_goal_binding"] = {}
    elif defect == "model":
        cem["model_after"] = {}
    elif defect == "roster":
        cem["nodes"]["s0_full_cem"][2]["boundary"] = 15
    elif defect == "cap":
        cem["nodes"]["s0_full_cem"][0]["receipt"]["account"]["charged_upper_bounds"]["predict_calls"] = 196
    else:
        structured["nodes"]["s0_structured"][0]["receipt"]["module_hook_audit"]["module_forward_pre_hooks"]["predict_calls"] = 3
    with pytest.raises(ValueError):
        audit.paired(structured, cem, 0)
