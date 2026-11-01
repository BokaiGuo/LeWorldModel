"""Non-neural orchestrator/algebra tests. No host GPU or model allocation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p4_receding_runtime_20261027_v1 as p4


def bank():
    return np.asarray([np.tile([i/20.,0.],(5,1)) for i in range(16)])


def frames():
    return [np.full((224,224,3),i,np.uint8) for i in range(3)]


def exercise(arm,abort=False):
    calls=[]
    events=[]
    decisions=[]
    tokens={"planning":0,"evaluation":0}
    physical=[]
    def encode(f,b):
        events.append("encode")
        calls.append((np.stack(f).copy(),b.copy()))
        return {"actual_boundary":int(f[-1][0,0,0]),"between":b.tolist(),"x":0.}
    def replay(context,actions,phase):
        tokens[phase]+=len(actions)//5
        # Compositional algebra, not a neural prediction.
        context=deepcopy(context)
        context["x"]+=float(np.asarray(actions)[:,0].sum())
        return context
    def metric(context):
        return abs(context["x"]-context["actual_boundary"])
    def seal(i,d):
        events.append("seal")
        decisions.append(deepcopy(d))
        return p4.content_sha(d)
    def execute(packet,i):
        events.append("execute")
        assert len(decisions)==i+1
        np.testing.assert_array_equal(packet,decisions[-1]["executed_packet"])
        physical.append(packet.copy())
        return {"status":"observation_unavailable" if abort else "completed",
            "completed_commands":2 if abort else 5,"remaining_commands_not_executed":3 if abort else 0},np.full((224,224,3),9,np.uint8)
    result=p4.branch_loop(arm=arm,root_index=0,bank=bank(),frames=frames(),between=np.zeros((2,5,2)),
        encode=encode,replay=replay,metric=metric,execute=execute,seal_decision=seal,
        journal=lambda i,e:None)
    return result,calls,decisions,tokens,physical,events


@pytest.mark.parametrize("arm",p4.ARMS)
def test_two_rounds_budget_and_pre_execution_seal(arm):
    r,observed,decisions,tokens,physical,events=exercise(arm)
    assert r["status"]=="two_rounds_completed" and r["physics_steps"]==10
    assert tokens=={"planning":160,"evaluation":4}
    assert events==["encode","seal","execute","encode","seal","execute"]
    assert len(observed)==2
    np.testing.assert_array_equal(observed[1][0],np.stack([frames()[1],frames()[2],np.full((224,224,3),9,np.uint8)]))
    np.testing.assert_array_equal(observed[1][1][-1],physical[0])
    assert decisions[1]["previous_decision_sha256"]==p4.content_sha(decisions[0])
    assert decisions[0]["observed"]["context_sha256"]!=decisions[1]["observed"]["context_sha256"]
    assert decisions[1]["observed_context_used_for_selection"] is (arm=="receding")
    if arm=="fixed":
        assert decisions[0]["selected"]==decisions[1]["selected"]
        assert decisions[0]["planning_context_sha256"]==decisions[1]["planning_context_sha256"]
        assert decisions[1]["executed_bank_index"]==decisions[0]["selected"]["path"][1]
    else:
        assert decisions[0]["planning_context_sha256"]!=decisions[1]["planning_context_sha256"]


@pytest.mark.parametrize("arm",p4.ARMS)
def test_abort_does_not_fake_context_or_budget_match(arm):
    r,obs,decisions,tokens,_,_=exercise(arm,abort=True)
    assert r["status"]=="guarded_abort" and r["physics_steps"]==2
    assert len(obs)==len(decisions)==1
    assert tokens["planning"]==(80 if arm=="receding" else 160)


def test_select_failure_keeps_all_candidates_and_refuses_action():
    journal=[]
    def fail(context,actions,phase):
        raise RuntimeError("injected prediction failure")
    with pytest.raises(RuntimeError,match="no deployable"):
        p4.select_pairs(context={},bank=bank(),paths=p4.paths_for(0)[:40],replay=fail,metric=lambda c:0.,journal=journal.append)
    assert len(journal)==40 and all(not r["completed"] for r in journal)


def test_failed_seal_never_dispatches():
    def seal(*args):
        raise OSError("storage fixture")
    def noexecute(*args):
        pytest.fail("physics reached before durable seal")
    with pytest.raises(OSError):
        p4.branch_loop(arm="fixed",root_index=0,bank=bank(),frames=frames(),between=np.zeros((2,5,2)),
            encode=lambda f,b:{"x":0},replay=lambda c,a,p:c,metric=lambda c:0.,execute=noexecute,
            seal_decision=seal,journal=lambda i,e:None)


def test_partial_history_refused():
    with pytest.raises(ValueError,match="partial/aborted"):
        p4.advance_observed(frames(),np.zeros((2,5,2)),frames()[0],bank()[0],
            {"status":"environment_terminated","completed_commands":5,"remaining_commands_not_executed":0})


def test_schedule_frozen_unique_and_shared_prefix():
    for i in range(4):
        paths=p4.paths_for(i)
        assert len(paths)==len(set(paths))==80
        assert paths==p4.paths_for(i)
        assert all(a in range(16) and b in range(16) for a,b in paths)
    assert p4.paths_for(0)!=p4.paths_for(1)


def test_cuda_guard_before_model_access():
    with pytest.raises(ValueError,match="CPU neural fallback"):
        p4.encode_cuda(None,SimpleNamespace(type="cpu"),None,None,None,None)


def test_actual_saved_p2_bank_is_preserved():
    path=Path("results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json")
    if not path.exists():
        pytest.skip("calibration artifact absent")
    for row in json.loads(path.read_text())["rows"]:
        b=[np.repeat(np.asarray(r["program_live_commands"][0])[None],5,axis=0) for r in row["compiled"]["receipts"]]
        np.testing.assert_array_equal(p4.validate_bank(b),b)


def test_finalize_source_failure_and_str_paths_always_report(tmp_path):
    input_path=tmp_path/"input.json"
    p4.write(input_path,{"real":"str binding key"})
    def sourcefail():
        raise ValueError("source mismatch")
    report=p4.finalize(tmp_path,[],{str(input_path):p4.digest(input_path)},sourcefail,lambda:True,{}, {})
    assert report["status"]=="integrity_fail"
    assert report["gates"]["inputs_unchanged"]
    assert not report["gates"]["sources_unchanged"]
    assert report["failures"][0]["message"]=="source mismatch"
    assert report["P4_admission"] is False
    assert report["scientific_status"]=="blocked_upstream"
    assert json.loads((tmp_path/"report.json").read_text())==report


def test_completed_but_wrong_budget_rejected(tmp_path):
    rows=[{"root_index":i,"arm":a,"status":"two_rounds_completed","predict_planning_samples":159,
        "predict_evaluation_samples":4,"image_encode_frames":6,"image_encode_calls":2} for i in range(4) for a in p4.ARMS]
    report=p4.finalize(tmp_path,rows,{},lambda:True,lambda:True,{}, {})
    assert not report["gates"]["completed_branches_actual_budget"]
    assert report["status"]=="integrity_fail"


def complete_rows():
    return [{"root_index":i,"arm":a,"status":"two_rounds_completed","predict_planning_samples":160,
        "predict_evaluation_samples":4,"predict_planning_calls":160,"predict_evaluation_calls":4,
        "action_encoder_work":{"calls":164,"samples":164},"image_encode_frames":6,"image_encode_calls":2,
        "predict":[{"completed":True,"failed_predict_calls":0}],"rounds":[{"execution":{"status":"completed"}}]}
        for i in range(4) for a in p4.ARMS]


def test_finalize_complete_is_only_engineering_never_scientific(tmp_path):
    report=p4.finalize(tmp_path,complete_rows(),{},lambda:True,lambda:True,{}, {})
    assert report["status"]=="engineering_calibration_recorded"
    assert report["P4_admission"] is report["efficacy_verified"] is False
    assert report["scientific_status"]=="blocked_upstream"
    assert report["full_two_round_branches"]==8


@pytest.mark.parametrize("fault",["calls","action_work","failed_predict","close_failure","executor_error"])
def test_forged_budget_or_executor_failure_cannot_pass(tmp_path,fault):
    rows=complete_rows()
    resources={}
    if fault=="calls":
        rows[0]["predict_planning_calls"]=159
    elif fault=="action_work":
        rows[0]["action_encoder_work"]["samples"]=163
    elif fault=="failed_predict":
        rows[0]["predict"][0]["failed_predict_calls"]=1
    elif fault=="close_failure":
        resources["setup_failure"]={"message":"env.close failed"}
    else:
        rows[0]["status"]="guarded_abort"
        rows[0]["rounds"][0]["execution"]["status"]="execution_failed"
    report=p4.finalize(tmp_path,rows,{},lambda:True,lambda:True,{},resources)
    assert report["status"]=="integrity_fail"
