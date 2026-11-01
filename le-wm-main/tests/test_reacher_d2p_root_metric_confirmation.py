"""Synthetic saved-array fixtures, never neural CPU fallback or efficacy."""
import copy

import pytest

from tests.test_analyze_reacher_d2p_masked_program_pilot import fixture as base_fixture
from experiments.spectral_reachability import analyze_reacher_d2p_root_metric_confirmation as analysis
from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta


@pytest.fixture
def fixture(base_fixture):
    report, predictions = copy.deepcopy(base_fixture)
    report["schema"] = "reacher_d2p_root_metric_confirmation.v1"
    report["gates"] = {k:True for k in analysis.REQUIRED_GATES}
    report["actual_predict_samples"] = 16830*24
    protocol = report["sealed_protocol"]
    protocol.update(policy="model-root-response-metric-v1",ridge=.01,goal_contract="joint-feasible-v2-soft-source",
        stage_terminal_thresholds_unchanged=True,all_forecasts_sealed_before_future_execution=True,
        excluded_source_episodes=list(range(224)))
    _,admission = feasible_goal_delta([0.,0.],[.65,-.65],[False,True],[[0.,0.],[-2.79,2.79]],allow_soft_source=True)
    for root in predictions["rows"]:
        root["goal_admission"] = admission
        for task,plan in root["tasks"].items():
            plan["actual_predict_samples"] = 16830
            plan["common_residual"] = plan["common_model_residual"]
            scored = []
            for b in plan["bank"]:
                b["response"] = [[0.]*(10*len(analysis.TASKS[task])) for _ in range(192)]
                s = {k:b[k] for k in ("index","prefix_cost","terminal_cost")}
                s["common_response_gain"] = {"bounded_quadratic_gain":0.,"bounded_quadratic_gain_upper_bound":0.,
                    "objective_certificate_passed":True,"physical_labels_used":False}
                scored.append(s)
            plan["scores"] = {"bank":scored,"decision":copy.deepcopy(plan["decision"]),
                "original_decision":copy.deepcopy(plan["decision"]),"certified_qps":17,
                "metric":{"device":"cuda:0","physical_labels_used":False}}
    return report,predictions


def test_complete_fresh_root_join_and_independent_fit_count(fixture):
    report, predictions = fixture
    predictions["rows"].reverse()
    result = analysis.summarize(report,predictions)
    assert result["independent_model_gain_fits"]==408
    assert not result["P1_admission"]
    assert len(result["rows"])==8
    assert all(a["retained_roots"]==8 for a in result["aggregate"].values())
    contrasts = result["aggregate"]["wrist_only"]["metric_minus_control"]["greedy"]
    assert "RGB" in contrasts["observed_terminal_cost"]["metric"]
    assert "joint-topology-aware" in contrasts["final_joint_distance"]["metric"]
    assert "fresh source-disjoint" in contrasts["final_joint_distance"]["metric"]


@pytest.mark.parametrize("bad",["source","partial","gain","changed_cost","mask","compute","oracle"])
def test_corrupt_fresh_evidence_rejected(fixture,bad):
    report,predictions = fixture
    plan = predictions["rows"][0]["tasks"]["shoulder_only"]
    if bad=="source": report["rows"][0]["source_episode"]=0
    elif bad=="partial": plan["scores"]["bank"].pop()
    elif bad=="gain": plan["scores"]["bank"][0]["common_response_gain"]["bounded_quadratic_gain_upper_bound"]=-1.
    elif bad=="changed_cost": plan["scores"]["bank"][0]["terminal_cost"]+=1.
    elif bad=="mask": report["rows"][0]["tasks"]["shoulder_only"]["bank_0"]["raw_actions"][0][1]=.1
    elif bad=="compute": plan["actual_predict_samples"]-=1
    else: plan["scores"]["metric"]["physical_labels_used"]=True
    with pytest.raises(ValueError): analysis.summarize(report,predictions)
