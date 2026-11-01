"""Saved-array numeric receipts only, not actual CUDA or efficacy claims."""
import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_compiler import (
    verify_spectrum,verify_fit,verify_native,branch_metrics,difference32,archive_analysis_sources,
    model_mechanism_diagnostics)
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import geometry
from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as native
from tests.test_reacher_d2p_proposal_native_selection import bank


def spectrum_fixture():
    response = np.zeros((192,10))
    response[:10] = np.diag(np.arange(10,0,-1))
    transform = np.eye(192)*.5
    rows = [{"raw_actions":np.zeros((15,2)).tolist()}]
    for amplitude in (.25,.75):
        for i in range(4):
            for sign in (1,-1):
                raw = np.zeros((15,2))
                raw.ravel()[i] = amplitude*sign
                rows.append({"raw_actions":raw.tolist()})
    return {"root_response":response.tolist(),"root_weighted_singular_values":(np.arange(10,0,-1)*.5).tolist(),"bank":rows},transform


def test_complete_spectral_action_witnesses():
    plan,transform = spectrum_fixture()
    verify_spectrum(plan,transform,"full_drive")


@pytest.mark.parametrize("defect",["zero","sign","direction","singular"])
def test_spectral_origin_corruption_fails(defect):
    plan,transform = spectrum_fixture()
    if defect == "zero": plan["bank"][0]["raw_actions"][0][0] = .1
    if defect == "sign": plan["bank"][2]["raw_actions"][0][0] = .25
    if defect == "direction":
        plan["bank"][1]["raw_actions"][0] = [0.,.25]
        plan["bank"][2]["raw_actions"][0] = [0.,-.25]
    if defect == "singular": plan["root_weighted_singular_values"][0] += .01
    with pytest.raises(ValueError): verify_spectrum(plan,transform,"full_drive")


def qp_fixture():
    fitted = geometry(np.eye(2),[.2,-.3])
    objective,baseline = fitted["regularized_objective"],fitted["baseline_squared_residual"]
    receipt = {"objective_certificate_passed":True,"physical_labels_used":False,
        "baseline_squared_residual":baseline,"bounded_quadratic_objective":objective,
        "bounded_quadratic_gain":baseline-objective,"bounded_quadratic_gain_upper_bound":baseline-objective,
        "qp_certificate":{"objective_duality_gap_upper_bound":0.,"active_set_terminated":True,
            "box_feasibility_violation":0.,"device":"cuda:0"}}
    return receipt,fitted


def test_independent_objective_and_gain_checked():
    receipt,fitted = qp_fixture()
    verify_fit(receipt,fitted)


@pytest.mark.parametrize("field",["bounded_quadratic_objective","bounded_quadratic_gain","baseline_squared_residual"])
def test_own_objective_or_gain_corruption_rejected(field):
    receipt,fitted = qp_fixture()
    receipt[field] += .01
    with pytest.raises(ValueError): verify_fit(receipt,fitted)


def test_native_decision_reconstructed_independently():
    rows = bank()
    for row in rows: row["terminal_cost"] = row["variants"]["v"]["terminal_cost"]
    decision = native.select(rows,"v")
    verify_native(rows,decision)
    changed = copy.deepcopy(decision)
    changed["native_index"] = (decision["native_index"]+1)%17
    with pytest.raises(ValueError): verify_native(rows,changed)


def physical_fixture():
    raw = np.zeros((15,2))
    observation = {"raw_actions":raw.tolist(),"packets":[{"packet":i+1,"observed_vector":[.1*(i+1)]*192,
        "qpos":[.1*(i+1),.2],"root_rgb_sha256":"0"*64,
        "action_sha256":hashlib.sha256(np.ascontiguousarray(raw[i*5:(i+1)*5],dtype=np.float32).tobytes()).hexdigest()}
        for i in range(3)]}
    plan = {"raw_actions":raw.tolist(),"terminal_vector":[0.]*192,"terminal_cost":0.}
    admission = {"feasible_target_qpos":[0.,0.],"limited_joints":[False,True]}
    return plan,observation,admission


def test_physical_task_and_proxy_metrics_remain_separate():
    plan,observation,admission = physical_fixture()
    result = branch_metrics(plan,observation,np.zeros(192),admission,"full_drive",np.eye(192)*.5,weighted=True)
    assert result["observed_terminal_cost"] == pytest.approx(.09)
    assert result["goal_Q_observed_terminal_cost"] == pytest.approx(.0225)
    assert result["final_joint_distance"] == pytest.approx(np.sqrt(.13))


def test_same_action_witness_corruption_rejected():
    plan,observation,admission = physical_fixture()
    observation["packets"][0]["action_sha256"] = "a"*64
    with pytest.raises(ValueError): branch_metrics(plan,observation,np.zeros(192),admission,"full_drive",np.eye(192),weighted=False)


def test_float32_residual_contract_is_reproduced_explicitly():
    left,right = [1.0000001],[1.]
    expected = (np.array(left,dtype=np.float32)-np.array(right,dtype=np.float32)).astype(np.float64)
    np.testing.assert_array_equal(difference32(left,right),expected)


def test_analysis_helpers_archived_without_overwrite(tmp_path):
    output = tmp_path/"analysis.json"
    hashes = archive_analysis_sources(output)
    assert set(hashes) == {"analyzer","physical_geometry","baseline_selector",
        "goal_receipt_and_contrasts","root_metric","checkpoint_contract","root_identity","goal_admission"}
    for key,sha in hashes.items():
        assert hashlib.sha256((output.with_suffix(".implementation")/(key+".py")).read_bytes()).hexdigest() == sha
    with pytest.raises(FileExistsError): archive_analysis_sources(output)


def test_model_rejection_audit_distinguishes_candidates_without_oracle_selection():
    rows = bank()
    rows[2]["prefix_cost"] = 1.1
    rows[2]["own_response_gain"]["bounded_quadratic_objective"] = 1.
    rows[2]["variants"]["v"]["terminal_cost"] = .1
    for row in rows: row["terminal_cost"] = row["variants"]["v"]["terminal_cost"]
    decision = native.select(rows,"v")
    result = model_mechanism_diagnostics(rows,decision)
    assert result["selected_is_greedy"] is False
    assert result["all_model_checks_count"] == 1
    assert result["programs"][2]["all_model_checks"] is True
    assert result["programs"][2]["stage_minus_greedy"] == pytest.approx(.6)
    assert result["physical_labels_used"] is False and result["selection_changed"] is False
    rows[2]["physical_outcome"] = -1e99
    assert model_mechanism_diagnostics(rows,decision) == result


def test_stage_penalty_can_reject_local_own_reach_improvement_without_code_failure():
    rows = bank()
    rows[2]["prefix_cost"] = 1.1
    rows[2]["own_response_gain"]["bounded_quadratic_objective"] = 180.
    rows[2]["variants"]["v"]["terminal_cost"] = .1
    for row in rows: row["terminal_cost"] = row["variants"]["v"]["terminal_cost"]
    result = model_mechanism_diagnostics(rows,native.select(rows,"v"))
    assert result["selected_is_greedy"] is True
    assert result["programs"][2]["checks"]["own_reach_better_than_controls"] is True
    assert result["programs"][2]["checks"]["total_score_better_than_controls"] is False
    assert result["all_model_checks_count"] == 0
