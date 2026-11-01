"""Explicit model/physical field access for two immutable capability sources.

No scientific report is rewritten or given a different schema. These getters
only reconcile the recorded action/prefix representation for exact replay.
"""
from __future__ import annotations

SUFFIX = "reacher_d2p_masked_suffix_search.v1"
METRIC = "reacher_d2p_root_metric_confirmation.v1"
FULL = "reacher_d2p_full_goal_metric_compiler.v1"


def task_plan(root, task, schema):
    if schema not in (SUFFIX,METRIC,FULL): raise ValueError("unsupported explicit capability reference schema")
    plan = root["tasks"][task]
    return plan["goal_metric"] if schema == FULL else plan


def physical_branch(root, task, index, schema):
    if schema not in (SUFFIX,METRIC,FULL): raise ValueError("unsupported explicit capability reference schema")
    branches = root["tasks"][task]
    if schema == FULL: branches = branches["goal_metric"]
    return branches["bank_"+str(index)]


def raw_actions(bank, schema):
    if schema == SUFFIX: return bank["variants"]["cem_then_gn4"]["raw_actions"]
    if schema in (METRIC,FULL): return bank["raw_actions"]
    raise ValueError("unsupported explicit capability reference schema")


def packets(branch, schema):
    if schema == SUFFIX: return branch["cem_then_gn4"]["packets"]
    if schema in (METRIC,FULL): return branch["packets"]
    raise ValueError("unsupported explicit capability reference schema")


def metric_roles(plan):
    bank = plan["bank"]
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("all17 ordered original model prefixes required")
    scores = plan["scores"]
    result = {"euclidean_candidate":scores["original_decision"]["native_index"],
        "metric_candidate":scores["decision"]["native_index"],
        "greedy":min(range(17), key=lambda i:(bank[i]["prefix_cost"],i)), "zero":0}
    if (any(type(i) is not int or not 0 <= i < 17 for i in result.values())
            or result["greedy"] != scores["decision"]["greedy_index"]
            or result["greedy"] != scores["original_decision"]["greedy_index"]
            or scores["decision"]["physical_labels_used"] is not False
            or scores["original_decision"]["physical_labels_used"] is not False):
        raise ValueError("complete model-only sealed metric/Euclidean/greedy/zero roles required")
    return result


def full_roles(plan):
    bank = plan["bank"]
    if len(bank) != 17 or [b["index"] for b in bank] != list(range(17)):
        raise ValueError("all17 full compiler prefixes required")
    native,common = plan["decision"],plan["historical_common_rule_decision"]
    result = {"native_candidate":native["native_index"],"common_candidate":common["native_index"],
        "greedy":min(range(17),key=lambda i:(bank[i]["prefix_cost"],i)),"zero":0}
    if (any(type(i) is not int or not 0 <= i < 17 for i in result.values())
            or any(d["greedy_index"] != result["greedy"] or d["physical_labels_used"] is not False for d in (native,common))):
        raise ValueError("unchanged native/common/greedy/zero model-only decisions required")
    return result
