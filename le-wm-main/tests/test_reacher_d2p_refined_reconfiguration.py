from copy import deepcopy

import pytest

from experiments.spectral_reachability.reacher_d2p_refined_reconfiguration import refine_witness_decision


def cells():
    return {"dispatch": {"immediate_goal_cost": 2., "refined_goal_cost": .1},
        "zero": {"immediate_goal_cost": 1., "refined_goal_cost": .3},
        "same_bank_greedy": {"immediate_goal_cost": .5, "refined_goal_cost": .2}}


def test_recheck_requires_non_greedy_and_both_refined_controls():
    decision = refine_witness_decision(cells(), had_witness=True)
    assert decision["witness_retained"] and decision["selected_branch"] == "dispatch"
    assert not decision["physical_labels_used"]


@pytest.mark.parametrize("control", ["zero", "same_bank_greedy"])
def test_equal_or_better_control_rejects_witness(control):
    value = cells()
    value[control]["refined_goal_cost"] = .1
    assert not refine_witness_decision(value, had_witness=True)["witness_retained"]


def test_absence_of_old_witness_cannot_be_promoted_by_recheck():
    decision = refine_witness_decision(cells(), had_witness=False)
    assert decision["selected_branch"] == "same_bank_greedy"


def test_immediate_non_greedy_contract_still_applies():
    value = cells()
    value["dispatch"]["immediate_goal_cost"] = .9
    assert not refine_witness_decision(value, had_witness=True)["witness_retained"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_costs_fail_closed(bad):
    value = cells()
    value["zero"]["refined_goal_cost"] = bad
    with pytest.raises(ValueError):
        refine_witness_decision(value, had_witness=True)


def test_does_not_mutate_model_inputs():
    value = cells()
    before = deepcopy(value)
    refine_witness_decision(value, had_witness=True)
    assert value == before
