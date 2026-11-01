"""Explicit source contracts; never physical-outcome-based role selection."""
import copy

import pytest

from experiments.spectral_reachability import reacher_d2p_capability_reference as source


def plan():
    return {"bank":[{"index":i,"prefix_cost":0. if i==4 else 1.,"raw_actions":[[i,0.]]} for i in range(17)],
        "scores":{"original_decision":{"native_index":4,"greedy_index":4,"physical_labels_used":False},
            "decision":{"native_index":12,"greedy_index":4,"physical_labels_used":False}}}


def test_all_model_roles_preserved_without_reading_physical_fields():
    p=plan()
    p["physical_outcomes"] = object()
    assert source.metric_roles(p)=={"euclidean_candidate":4,"metric_candidate":12,"greedy":4,"zero":0}
    assert source.raw_actions(p["bank"][12],source.METRIC)==[[12,0.]]
    assert source.packets({"packets":["complete receipt"]},source.METRIC)==["complete receipt"]


def test_old_source_accessor_preserves_original_nested_contract():
    raw=[[.1,0.]]
    assert source.raw_actions({"variants":{"cem_then_gn4":{"raw_actions":raw}}},source.SUFFIX) is raw
    packets=[{"packet":1}]
    assert source.packets({"cem_then_gn4":{"packets":packets}},source.SUFFIX) is packets


@pytest.mark.parametrize("bad",["count","order","bool_index","greedy","oracle"])
def test_incomplete_or_unsealed_roles_rejected(bad):
    p=copy.deepcopy(plan())
    if bad=="count": p["bank"].pop()
    elif bad=="order": p["bank"].reverse()
    elif bad=="bool_index": p["scores"]["decision"]["native_index"]=True
    elif bad=="greedy": p["scores"]["decision"]["greedy_index"]=1
    else: p["scores"]["decision"]["physical_labels_used"]=True
    with pytest.raises(ValueError): source.metric_roles(p)


def test_unsupported_schema_cannot_be_silently_reinterpreted():
    with pytest.raises(ValueError): source.raw_actions({},"synthetic")
    with pytest.raises(ValueError): source.packets({},"synthetic")
    with pytest.raises(ValueError): source.task_plan({},"full_drive","synthetic")
    with pytest.raises(ValueError): source.physical_branch({},"full_drive",0,"synthetic")


def full_plan():
    p = plan()
    p["decision"] = p["scores"]["decision"]
    p["historical_common_rule_decision"] = p["scores"]["original_decision"]
    return p


def test_full_compiler_uses_own_nested_source_without_rewriting_report():
    p = full_plan()
    root = {"tasks":{"full_drive":{"original":{"unrelated":True},"goal_metric":p}}}
    assert source.task_plan(root,"full_drive",source.FULL) is p
    assert source.full_roles(p) == {"native_candidate":12,"common_candidate":4,"greedy":4,"zero":0}
    assert source.raw_actions(p["bank"][12],source.FULL) == [[12,0.]]
    branch = {"packets":[{"packet":1}]}
    root = {"tasks":{"full_drive":{"goal_metric":{"bank_12":branch}}}}
    assert source.physical_branch(root,"full_drive",12,source.FULL) is branch
    assert source.packets(branch,source.FULL) is branch["packets"]


@pytest.mark.parametrize("defect",["partial","greedy","oracle","bool_index"])
def test_full_compiler_roles_are_strictly_model_sealed(defect):
    p = full_plan()
    if defect == "partial": p["bank"].pop()
    elif defect == "greedy": p["decision"]["greedy_index"] = 0
    elif defect == "oracle": p["decision"]["physical_labels_used"] = True
    else: p["decision"]["native_index"] = True
    with pytest.raises(ValueError): source.full_roles(p)
