"""Synthetic contract tests, not neural or physical efficacy evidence."""
import copy

import pytest

from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as selector


def bank():
    return [{"index":i,"prefix_cost":1. if i != 1 else .5,
        "own_response_gain":{"objective_certificate_passed":True,"physical_labels_used":False,
            "bounded_quadratic_objective":192.,"baseline_squared_residual":200.,
            "qp_certificate":{"device":"cuda:0","active_set_terminated":True,
                "box_feasibility_violation":0.,"objective_duality_gap_upper_bound":0.}},
        "variants":{"v":{"terminal_cost":1.}}} for i in range(17)]


def test_absolute_own_objective_not_common_gain_or_greedy_priority():
    rows = bank()
    rows[2]["prefix_cost"] = 1.1
    rows[2]["own_response_gain"]["bounded_quadratic_objective"] = 1.
    rows[2]["variants"]["v"]["terminal_cost"] = .1
    result = selector.select(rows,"v")
    assert result["native_index"] == 2 and result["greedy_index"] == 1
    assert result["reconfiguration_witness_found"] is True
    assert result["total_score_upper"][2] == 1.1+1/192
    assert result["physical_labels_used"] is False and result["P1_admission"] is False


def test_progress_is_selected_without_manufacturing_a_non_greedy_witness():
    result = selector.select(bank(),"v")
    assert result["native_index"] == 1 and result["reconfiguration_witness_found"] is False


def test_terminal_value_does_not_override_proposal_objective_or_use_outcomes():
    rows = bank()
    rows[2]["variants"]["v"]["terminal_cost"] = 0.
    rows[2]["observed_physical_cost"] = -1e9
    assert selector.select(rows,"v")["native_index"] == 1


def test_equal_total_scores_use_original_index():
    rows = bank()
    rows[0]["prefix_cost"] = .5
    assert selector.select(rows,"v")["native_index"] == 0


@pytest.mark.parametrize("mutation",["partial","duplicate","nan","oracle","device","uncertified","gap","infeasible","negative"])
def test_invalid_or_non_cuda_evidence_rejected(mutation):
    rows = copy.deepcopy(bank())
    gain = rows[0]["own_response_gain"]
    if mutation == "partial": rows.pop()
    elif mutation == "duplicate": rows[1]["index"] = 0
    elif mutation == "nan": rows[0]["prefix_cost"] = float("nan")
    elif mutation == "oracle": gain["physical_labels_used"] = True
    elif mutation == "device": gain["qp_certificate"]["device"] = "cpu"
    elif mutation == "uncertified": gain["objective_certificate_passed"] = False
    elif mutation == "gap": gain["qp_certificate"]["objective_duality_gap_upper_bound"] = .1
    elif mutation == "infeasible": gain["qp_certificate"]["box_feasibility_violation"] = .01
    else: gain["bounded_quadratic_objective"] = -1.
    with pytest.raises(ValueError): selector.select(rows,"v")
