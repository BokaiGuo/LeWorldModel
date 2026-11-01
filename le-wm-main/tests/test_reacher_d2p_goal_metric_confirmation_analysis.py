"""Pure numeric and receipt fixtures, never neural CPU execution."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation import verify_goal_receipt, contrast, check_scores


def receipt():
    return {"encoder_forwards":13,"input_vjp_rows":192,"same_metric_all_tasks":True,
        "physical_future_labels_read":False,"gram":np.eye(192).tolist(),
        "gradient_checks":[{"direction":i,"step":s,"relative_error":.001,"cosine":.999}
            for i in range(3) for s in (.001,.003)]}


def test_receipt_and_gram_reconstructed_without_mutation():
    data = receipt()
    saved = copy.deepcopy(data)
    transform = verify_goal_receipt(data)
    np.testing.assert_allclose(transform @ transform,np.eye(192)*(.01/1.01),atol=1e-14)
    assert data == saved


@pytest.mark.parametrize("defect",["forward","vjp","labels","task","missing_fd","fd_error","fd_cosine","nonfinite","gram"])
def test_corrupt_receipts_fail_closed(defect):
    data = receipt()
    if defect == "forward": data["encoder_forwards"] = 12
    if defect == "vjp": data["input_vjp_rows"] = 191
    if defect == "labels": data["physical_future_labels_read"] = True
    if defect == "task": data["same_metric_all_tasks"] = False
    if defect == "missing_fd": data["gradient_checks"].pop()
    if defect == "fd_error": data["gradient_checks"][0]["relative_error"] = .151
    if defect == "fd_cosine": data["gradient_checks"][0]["cosine"] = .979
    if defect == "nonfinite": data["gradient_checks"][0]["cosine"] = float("nan")
    if defect == "gram": data["gram"][0][0] = -1.
    with pytest.raises(ValueError): verify_goal_receipt(data)


def test_all_roots_contrast_retains_harm_and_labels_direction():
    summary = contrast([-1.,-1.,-1.,-1.,-1.,-1.,-1.,8.])
    assert summary["wins"] == 7 and summary["losses"] == 1
    assert summary["mean"] == .125
    assert "negative better" in summary["scope"]


def score_fixture():
    import torch
    from experiments.spectral_reachability.bounded_response_qp import certified_response_gain
    from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
    from experiments.spectral_reachability.reacher_d2p_goal_metric_runtime import score_bank
    response = torch.zeros((192,10),dtype=torch.float64)
    response[:10] = torch.eye(10,dtype=torch.float64)
    residual = torch.full((192,),.2,dtype=torch.float64)
    gain = certified_response_gain(response,residual,damping=.01)
    plan = {"common_residual":residual.tolist(),"bank":[{"index":i,
        "prefix_cost":.2+i/100.,"terminal_cost":.3+i/100.,
        "prefix_vector":[.1+i/100.]*192,"terminal_vector":[.2+i/100.]*192,
        "response":response.tolist(),"common_response_gain":gain} for i in range(17)]}
    metric = GoalEncoderMetric(torch.eye(192,dtype=torch.float64))
    target = np.full((192,),.4)
    plan["goal_scores"] = score_bank(metric,plan,target)
    return plan,target,metric.sqrt_matrix.numpy()


def test_all17_scores_independently_reconstructed():
    plan,target,transform = score_fixture()
    assert check_scores(plan,target,transform,width=10) == 17


@pytest.mark.parametrize("defect",["cost","gain","oracle","decision","partial"])
def test_saved_score_corruption_rejected(defect):
    plan,target,transform = score_fixture()
    original = plan["goal_scores"]["gain_only_original_costs"]
    weighted = plan["goal_scores"]["consistent_Q_costs"]
    if defect == "cost": weighted["bank"][0]["terminal_cost"] += .01
    if defect == "gain": original["bank"][0]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] = -1.
    if defect == "oracle": original["bank"][0]["common_response_gain"]["physical_labels_used"] = True
    if defect == "decision": weighted["decision"]["native_index"] = (weighted["decision"]["native_index"]+1)%17
    if defect == "partial": weighted["bank"].pop()
    with pytest.raises(ValueError): check_scores(plan,target,transform,width=10)
