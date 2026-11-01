"""Synthetic receipts check data contracts, not CUDA or scientific efficacy."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_masked_program_pilot as analysis
from tests.test_analyze_reacher_d2p_masked_task_preflight import endpoint
from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta


@pytest.fixture
def fixture():
    target = np.full(192,.1)
    _,admission = feasible_goal_delta([0.,0.],[.65,-.65],[False,True],[[0.,0.],[-2.79,2.79]])
    plans,measurements = {},{}
    for task in analysis.TASKS:
        bank,actual = [],{}
        for i in range(17):
            raw = np.zeros((15,2),np.float32)
            if i: raw[:5,list(analysis.TASKS[task])] = .01*i
            prefix = np.full(192,.01*i)
            value = .1 if i else .09
            terminal = np.full(192,value)
            branch = {"index":i,"raw_actions":raw.tolist(),"prefix_vector":prefix.tolist(),"terminal_vector":terminal.tolist(),
                "prefix_cost":float(np.mean((prefix-target)**2)),"terminal_cost":float(np.mean((terminal-target)**2)),"full_split_exact":True,
                "common_response_gain":{"bounded_quadratic_gain":.1,"bounded_quadratic_gain_upper_bound":.1,
                    "bounded_quadratic_objective":1.82,"baseline_squared_residual":1.92,
                    "objective_certificate_passed":True,"physical_labels_used":False,
                    "qp_certificate":{"active_set_terminated":True,"box_feasibility_violation":0.,"device":"cuda:0","objective_duality_gap_upper_bound":0.}},
                "suffix_receipt":{"trace":[{"cost_before":.1,"cost_after":.01} for _ in range(4)]}}
            bank.append(branch)
            packets = []
            for j in range(3):
                point = endpoint(raw[j*5:(j+1)*5])
                packets.append({"packet":j+1,"observed_vector":point["observed_vector"],"qpos":point["qpos"],**point["boundaries"][0]})
            actual["bank_"+str(i)] = {"raw_actions":raw.tolist(),"packets":packets}
        controls = {name:copy.deepcopy(bank[1]) for name in analysis.CONTROLS}
        actual.update({name:copy.deepcopy(actual["bank_1"]) for name in analysis.CONTROLS})
        greedy,native,witnesses = analysis.independent_decision(bank)
        plans[task] = {"bank":bank,"controls":controls,"trained_modules":False,"compute_matched":False,
            "actual_predict_samples":2482,"common_model_residual":[.1]*192,
            "decision":{"greedy_index":greedy,"native_index":native,"qualified_indices":witnesses,
                "reconfiguration_witness_found":bool(witnesses),"physical_labels_used":False}}
        measurements[task] = actual
    roots = [{"seed":i,"source_episode":300+i,"source_start_row":1000*i} for i in range(8)]
    protocol = {"root_seeds":list(range(8)),"source_roots":roots,"excluded_source_episodes":list(range(200)),
        "tasks":{k:list(v) for k,v in analysis.TASKS.items()},"prefix_packets":1,"future_packets":2,
        "spectral_bank":{"directions":4,"amplitudes":[.25,.75],"signs":[-1,1],"literal_zero_index":0,"count":17},
        "suffix_gn_iterations":4,"flat_gn_iterations":4,"damping":.01,"line_search":[1.,.5,.25,.125,.0625,.03125],
        "non_greedy_margin":1e-6,"terminal_margin":1e-6,"goal_contract":"joint-feasible-v1","target_delta":[.65,-.65],
        "no_post_training":True,"all_predictions_sealed_before_any_future_execution":True,"compute_matched":False,"scope":"synthetic fixture"}
    forecast = [{**r,"goal_vector":target.tolist(),"goal_admission":copy.deepcopy(admission),"tasks":copy.deepcopy(plans)} for r in roots]
    measured = [{**r,"tasks":copy.deepcopy(measurements)} for r in roots]
    report = {"schema":"reacher_d2p_masked_program_pilot.v1","status":"completed","sealed_protocol":protocol,
        "gates":{k:True for k in analysis.REQUIRED_GATES},"rows":measured,"physical_branches":456,"actual_model_predict_samples":59568}
    return report,{"rows":forecast}


def test_complete_receipt_and_root_identity_join(fixture):
    report,predictions = fixture
    predictions["rows"].reverse()
    result = analysis.summarize(report,predictions)
    assert len(result["rows"]) == 8 and result["P1_admission"] is False
    assert result["physical_branches"] == 456
    assert set(result["aggregate"]) == set(analysis.TASKS)


@pytest.mark.parametrize("mutation",["gate","duplicate","source","task","action","boundary","vector","cost","selection","oracle","trace","split","compute","zero","gain"])
def test_partial_or_changed_evidence_rejected(fixture,mutation):
    report,predictions = fixture
    plan = predictions["rows"][0]["tasks"]["shoulder_only"]
    measured = report["rows"][0]["tasks"]["shoulder_only"]
    if mutation == "gate": report["gates"]["cuda_only"] = 1
    elif mutation == "duplicate": report["rows"][1]["seed"] = 0
    elif mutation == "source": predictions["rows"][0]["source_episode"] = 999
    elif mutation == "task": measured.pop("flat_gn")
    elif mutation == "action": measured["bank_0"]["raw_actions"][0][1] = .1
    elif mutation == "boundary": measured["bank_0"]["packets"][0]["action_sha256"] = "b"*64
    elif mutation == "vector": measured["bank_0"]["packets"][0]["observed_vector"] = [0.,0.]
    elif mutation == "cost": plan["bank"][0]["terminal_cost"] += .1
    elif mutation == "selection": plan["decision"]["native_index"] = 16
    elif mutation == "oracle": plan["bank"][0]["common_response_gain"]["physical_labels_used"] = True
    elif mutation == "trace": plan["bank"][0]["suffix_receipt"]["trace"].pop()
    elif mutation == "split": plan["bank"][0]["full_split_exact"] = False
    elif mutation == "compute": plan["actual_predict_samples"] -= 1
    elif mutation == "zero": plan["bank"][0]["raw_actions"][0][0] = .1
    else: plan["bank"][0]["common_response_gain"]["bounded_quadratic_gain_upper_bound"] += 1
    with pytest.raises(ValueError): analysis.summarize(report,predictions)


def test_declared_greedy_fallback_does_not_become_a_witness(fixture):
    result = analysis.summarize(*fixture)
    for task in result["aggregate"].values():
        assert task["model_witness_roots"] == 0
        assert task["witness_roots_observed_non_greedy"] == 0


def test_further_fresh_confirmation_can_exclude_more_than_original200(fixture):
    report,predictions = fixture
    report["sealed_protocol"]["excluded_source_episodes"] = list(range(208))
    result = analysis.summarize(report,predictions)
    assert len(result["rows"]) == 8


@pytest.mark.parametrize("excluded",[list(range(199)),list(range(208))+[300]])
def test_too_few_or_overlapping_exclusions_rejected(fixture,excluded):
    report,predictions = fixture
    report["sealed_protocol"]["excluded_source_episodes"] = excluded
    with pytest.raises(ValueError): analysis.summarize(report,predictions)
