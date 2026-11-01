"""Synthetic audit receipts, never empirical CUDA/physics evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_masked_suffix_search as analysis
from tests.test_analyze_reacher_d2p_masked_program_pilot import fixture as baseline_fixture


def receipt(response,residual):
    # This fixture uses a zero response, so the exact objective is ||r||².
    assert np.count_nonzero(response)==0
    baseline = float(residual@residual)
    return {"objective_certificate_passed":True,"physical_labels_used":False,"bounded_quadratic_gain":0.,
        "bounded_quadratic_objective":baseline,"baseline_squared_residual":baseline,
        "bounded_quadratic_gain_upper_bound":0.,"response_singular_values":[0.]*response.shape[1],
        "qp_certificate":{"device":"cuda:0","active_set_terminated":True,"box_feasibility_violation":0.,"objective_duality_gap_upper_bound":0.}}


@pytest.fixture
def fixture(baseline_fixture):
    original,old_predictions = baseline_fixture
    rows,forecasts = [],[]
    for old,p in zip(original["rows"],old_predictions["rows"]):
        task_plans,task_observations = {},{}
        for task in analysis.TASKS:
            old_task = p["tasks"][task]
            common = np.asarray(old_task["common_model_residual"])
            response = np.zeros((192,len(analysis.columns(task,2))))
            gain = receipt(response,common)
            bank,observations = [],{}
            for b in old_task["bank"]:
                variants = {v:{k:copy.deepcopy(b[k]) for k in ("raw_actions","terminal_vector","terminal_cost","full_split_exact")} for v in analysis.VARIANTS}
                bank.append({"index":b["index"],"prefix_vector":b["prefix_vector"],"prefix_cost":b["prefix_cost"],
                    "zero_suffix_vector":[0.]*192,"own_residual":np.asarray(p["goal_vector"],np.float32).astype(float).tolist(),
                    "response":response.tolist(),"common_response_gain":copy.deepcopy(gain),
                    "own_response_gain":receipt(response,np.asarray(p["goal_vector"],np.float32).astype(float)),"variants":variants,
                    "cem_receipt":{"seed":p["seed"]+100000+b["index"],"samples":128,"steps":3,"elite":16,
                        "model_terminal_sample_evaluations":386,"physical_labels_used":False},
                    "gn_receipt":{"physical_labels_used":False,"trace":[{"cost_before":.1,"cost_after":.01} for _ in range(4)]}})
                observations["bank_"+str(b["index"])] = {v:copy.deepcopy(old["tasks"][task]["bank_"+str(b["index"])]) for v in analysis.VARIANTS}
            task_plans[task] = {"bank":bank,"decisions":{v:copy.deepcopy(old_task["decision"]) for v in analysis.VARIANTS},
                "common_residual":common.tolist(),"actual_predict_samples":14555}
            task_observations[task] = observations
        forecasts.append({**{k:copy.deepcopy(p[k]) for k in ("seed","source_episode","source_start_row","goal_vector","goal_admission")},
            "root_rgb_sha256":"a"*64,"goal_rgb_sha256":"b"*64,"initial_snapshot":{"synthetic":True},"tasks":task_plans})
        p.update(root_rgb_sha256="a"*64,goal_rgb_sha256="b"*64,initial_snapshot={"synthetic":True})
        rows.append({**{k:old[k] for k in ("seed","source_episode","source_start_row")},"tasks":task_observations})
    protocol = {"root_seeds":list(range(8)),"tasks":{k:list(v) for k,v in analysis.TASKS.items()},
        "variants":list(analysis.VARIANTS),"prefix_packets":1,"future_packets":2,"bank_count":17,
        "cem":{"samples":128,"steps":3,"elite":16,"seed_law":"root seed + 100000 + prefix index","initial":"same fixed old GN4 suffix"},
        "gn_after_cem_iterations":4,"damping":.01,"prefixes_goals_thresholds_unchanged":True,
        "all_predictions_sealed_before_any_future_execution":True,"no_post_training":True,"compute_matched":False,"scope":"synthetic receipt"}
    report = {"schema":"reacher_d2p_masked_suffix_search.v1","status":"completed","gates":{k:True for k in analysis.REQUIRED_GATES},
        "sealed_protocol":protocol,"physical_branches":1224,"actual_model_predict_samples":349320,"rows":rows}
    return report,{"rows":forecasts},original,old_predictions


def test_all_full_response_solves_and_identity_join(fixture):
    report,predictions,original,old_predictions = fixture
    predictions["rows"].reverse()
    original["rows"].reverse()
    result = analysis.summarize(report,predictions,original,old_predictions)
    assert result["independent_scipy_geometry_solves"]==816
    assert result["physical_branches"]==1224 and result["P1_admission"] is False
    assert len(result["rows"])==8


@pytest.mark.parametrize("mutation",["gate","partial","duplicate","source","goal","matrix","own","common","gain","variant",
    "prefix","replay","decision","steps","seed","accounting","task_compute","model_cost"])
def test_evidence_changes_rejected(fixture,mutation):
    report,predictions,original,old_predictions = fixture
    plan = predictions["rows"][0]["tasks"]["full_drive"]
    measured = report["rows"][0]["tasks"]["full_drive"]
    b = plan["bank"][0]
    if mutation=="gate": report["gates"]["cuda_only"]=1
    elif mutation=="partial": report["rows"].pop()
    elif mutation=="duplicate": predictions["rows"][1]["seed"]=0
    elif mutation=="source": report["rows"][0]["source_start_row"]+=1
    elif mutation=="goal": predictions["rows"][0]["goal_vector"][0]+=.1
    elif mutation=="matrix": b["response"][0][0]=1.
    elif mutation=="own": b["own_residual"][0]+=.1
    elif mutation=="common": plan["common_residual"][0]+=.1
    elif mutation=="gain": b["own_response_gain"]["bounded_quadratic_gain"]+=.1
    elif mutation=="variant": measured["bank_0"].pop("cem_then_gn4")
    elif mutation=="prefix": b["variants"]["cem_incumbent"]["raw_actions"][0][0]=.1
    elif mutation=="replay": measured["bank_0"]["old_gn4"]["packets"][0]["observed_vector"][0]+=.1
    elif mutation=="decision": plan["decisions"]["cem_incumbent"]["native_index"]=16
    elif mutation=="steps": b["gn_receipt"]["trace"].pop()
    elif mutation=="seed": b["cem_receipt"]["seed"]+=1
    elif mutation=="accounting": report["actual_model_predict_samples"]-=1
    elif mutation=="task_compute": plan["actual_predict_samples"]-=1
    else: b["variants"]["cem_incumbent"]["terminal_cost"]+=1
    with pytest.raises(ValueError): analysis.summarize(report,predictions,original,old_predictions)


def test_nonzero_response_geometry_independently_validates_boundary_solution():
    response,residual = np.eye(2),np.array([3.,.1])
    action = np.array([1.,.1/1.01])
    objective = float(np.sum((residual-response@action)**2)+.01*np.sum(action**2))
    baseline = float(residual@residual)
    value = {"objective_certificate_passed":True,"physical_labels_used":False,"bounded_quadratic_gain":baseline-objective,
        "bounded_quadratic_objective":objective,"baseline_squared_residual":baseline,"response_singular_values":[1.,1.],
        "qp_certificate":{"active_set_terminated":True,"box_feasibility_violation":0.,"device":"cuda","objective_duality_gap_upper_bound":0.}}
    result = analysis.verify_gain(value,response,residual)
    np.testing.assert_allclose(result["bounded_objective"],objective,atol=1e-12)


def enable_proposal(report,predictions):
    from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as selection
    report["sealed_protocol"]["proposal_native_contract"] = copy.deepcopy(selection.CONTRACT)
    for r in predictions["rows"]:
        for p in r["tasks"].values():
            p["proposal_native_decisions"] = {v:selection.select(p["bank"],v) for v in analysis.VARIANTS}


def test_presealed_proposal_formula_checked_independently(fixture):
    report,predictions,original,old_predictions = fixture
    enable_proposal(report,predictions)
    result = analysis.summarize(*fixture)
    assert result["proposal_native_presealed"] is True
    for a in result["aggregate"].values():
        assert set(a["proposal_native"]["model_witness_roots"]) == set(analysis.VARIANTS)


@pytest.mark.parametrize("mutation",["beta","missing","unsealed","decision","score","oracle","fresh_exclusion"])
def test_proposal_or_confirmation_protocol_tampering_rejected(fixture,mutation):
    report,predictions,original,old_predictions = fixture
    enable_proposal(report,predictions)
    p = predictions["rows"][0]["tasks"]["full_drive"]
    if mutation == "beta": report["sealed_protocol"]["proposal_native_contract"]["beta"] = 2.
    elif mutation == "missing": p["proposal_native_decisions"].pop("old_gn4")
    elif mutation == "unsealed": report["sealed_protocol"].pop("proposal_native_contract")
    elif mutation == "decision": p["proposal_native_decisions"]["old_gn4"]["native_index"] = 16
    elif mutation == "score": p["proposal_native_decisions"]["old_gn4"]["total_score_upper"][0] += 1.
    elif mutation == "oracle": p["proposal_native_decisions"]["old_gn4"]["physical_labels_used"] = True
    else:
        report["sealed_protocol"].update(source_disjoint_confirmation=True,excluded_source_episodes=[])
    with pytest.raises(ValueError): analysis.summarize(*fixture)


@pytest.mark.parametrize("mutation",["negative","nan","upper"])
def test_geometry_certificate_bounds_are_finite_and_consistent(mutation):
    response,residual = np.zeros((2,2)),np.ones(2)
    value = receipt(response,residual)
    if mutation=="upper": value["bounded_quadratic_gain_upper_bound"] = 1.
    else: value["qp_certificate"]["objective_duality_gap_upper_bound"] = -1e-12 if mutation=="negative" else float("nan")
    with pytest.raises(ValueError): analysis.verify_gain(value,response,residual)
