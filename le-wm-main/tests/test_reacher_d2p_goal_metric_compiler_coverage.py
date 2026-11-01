"""Numeric audit fixtures, not physical-control or neural evidence."""
import copy

import pytest

from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as native
from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_compiler import model_mechanism_diagnostics
from experiments.spectral_reachability.audit_reacher_d2p_goal_metric_compiler_coverage import diagnose_task
from tests.test_reacher_d2p_proposal_native_selection import bank


def fixture():
    rows = bank()
    for row in rows: row["terminal_cost"] = row["variants"]["v"]["terminal_cost"]
    decision = native.select(rows,"v")
    metrics = {"bank_"+str(i):{"prefix_joint_distance":1.,"final_joint_distance":1.,
        "observed_prefix_cost":1.,"observed_terminal_cost":1.,
        "goal_Q_observed_prefix_cost":1.,"goal_Q_observed_terminal_cost":1.} for i in range(17)}
    metrics.update({"flat_gn":{},"flat_cem_gn_incumbent":{}})
    observed = {"decision":decision,"model_mechanism_diagnostics":model_mechanism_diagnostics(rows,decision),
        "all_bank_metrics":{"goal_metric":metrics}}
    return {"bank":rows,"decision":decision},observed


def test_physical_candidate_does_not_become_a_deployed_selection():
    plan,observed = fixture()
    actual = observed["all_bank_metrics"]["goal_metric"]["bank_2"]
    actual.update(prefix_joint_distance=2.,final_joint_distance=.2,
        observed_prefix_cost=2.,observed_terminal_cost=.2)
    result = diagnose_task(plan,observed)
    assert result["counts"]["joint_and_raw_rgb_stage_terminal"] == 1
    assert result["selected_index"] == 1
    assert result["complete_model_candidates"] == 0
    assert result["physical_capability_measured"] is False and result["selection_changed"] is False


def test_same_Q_prediction_failure_separate_from_task_proxy_failure():
    plan,observed = fixture()
    actual = observed["all_bank_metrics"]["goal_metric"]["bank_2"]
    actual.update(goal_Q_observed_prefix_cost=2.,goal_Q_observed_terminal_cost=.2)
    result = diagnose_task(plan,observed)
    checks = result["programs"][2]["checks"]
    assert checks["model_Q_vs_actual_Q_stage_disagreement"] is True
    assert checks["actual_Q_vs_joint_stage_disagreement"] is True
    assert checks["joint_and_goal_Q_stage_terminal"] is False


@pytest.mark.parametrize("defect",["missing","decision","diagnosis"])
def test_incomplete_or_drifted_contract_rejected(defect):
    plan,observed = fixture()
    if defect == "missing": observed["all_bank_metrics"]["goal_metric"].pop("bank_16")
    elif defect == "decision": observed["decision"] = {**plan["decision"],"native_index":2}
    else:
        observed["model_mechanism_diagnostics"] = copy.deepcopy(observed["model_mechanism_diagnostics"])
        observed["model_mechanism_diagnostics"]["selection_changed"] = True
    with pytest.raises(ValueError): diagnose_task(plan,observed)
