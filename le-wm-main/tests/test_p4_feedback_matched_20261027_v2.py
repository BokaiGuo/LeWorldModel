"""NON-NEURAL algebra/metadata fakes only; no CUDA, model or training."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p4_feedback_matched_20261027_v2 as p4


def bank():
    return np.asarray([np.tile([i/20.,0.],(5,1)) for i in range(16)])


def frames():
    return [np.full((224,224,3),i,np.uint8) for i in range(3)]


def exercise(arm, *, abort_round=None, terminal_value=9, fail_seal=False, bad_replay=False):
    observed, decisions, journal, physical, calls, order = [], [], [], [], [], []
    def encode(f,b):
        order.append("encode")
        observed.append((np.stack(f).copy(),b.copy()))
        return {"boundary":int(f[-1][0,0,0]),"x":0.,"between":b.tolist()}
    def replay(context, actions, phase):
        calls.append({"round":len(observed)-1,"phase":phase,"tokens":len(actions)//5,
                      "context":deepcopy(context)})
        result=deepcopy(context)
        # Non-neural additive algebra, NOT a predictive-efficacy fixture.
        result["x"]+=float(np.asarray(actions)[:,0].sum())
        if bad_replay and phase=="evaluation":
            result["x"]+=1
        return result
    def seal(i,d):
        order.append("seal")
        if fail_seal:
            raise OSError("injected storage failure")
        decisions.append(deepcopy(d))
        return p4.content_sha(d)
    def execute(packet,i):
        order.append("execute")
        assert len(decisions)==i+1
        np.testing.assert_array_equal(packet,decisions[-1]["executed_packet"])
        physical.append(packet.copy())
        abort=i==abort_round
        return {"status":"observation_unavailable" if abort else "completed",
                "completed_commands":2 if abort else 5,
                "remaining_commands_not_executed":3 if abort else 0},np.full(
                    (224,224,3),terminal_value,np.uint8)
    result=p4.branch_loop(arm=arm,root_index=0,bank=bank(),frames=frames(),
        between=np.zeros((2,5,2)),encode=encode,replay=replay,
        metric=lambda c:abs(c["x"]-c["boundary"]),execute=execute,
        seal_decision=seal,journal=lambda i,e:journal.append({"round":i,**e}))
    return SimpleNamespace(result=result,observed=observed,decisions=decisions,
        journal=journal,physical=physical,calls=calls,order=order)


@pytest.mark.parametrize("arm",p4.ARMS)
def test_equal_per_round_budget_schedule_and_seal_before_physics(arm):
    x=exercise(arm)
    assert x.result["status"]=="two_rounds_completed"
    assert x.result["physics_steps"]==10
    assert x.order==["encode","seal","execute","encode","seal","execute"]
    for i in range(2):
        group=[r for r in x.calls if r["round"]==i]
        assert [r["tokens"] for r in group if r["phase"]=="planning"]==[2]*40
        assert [r["tokens"] for r in group if r["phase"]=="evaluation"]==[2,1,1]
        assert [r["path"] for r in x.journal if r["round"]==i]==[
            list(p) for p in p4.paths_for(0)[i*40:(i+1)*40]]
        assert x.decisions[i]["executed_bank_index"]==x.decisions[i]["selected"]["path"][0]
    assert x.decisions[1]["previous_decision_sha256"]==p4.content_sha(x.decisions[0])


def test_two_arms_identical_initial_decision_and_all_schedules():
    a,b=exercise("actual"),exercise("imagined")
    assert a.decisions[0]["selected"]==b.decisions[0]["selected"]
    for i in range(2):
        assert a.decisions[i]["schedule"]==b.decisions[i]["schedule"]
        assert a.decisions[i]["search_horizon_tokens"]==b.decisions[i]["search_horizon_tokens"]==2


@pytest.mark.parametrize("arm",p4.ARMS)
def test_observed_history_shift_is_real_and_exact(arm):
    x=exercise(arm)
    np.testing.assert_array_equal(x.observed[1][0],np.stack([
        frames()[1],frames()[2],np.full((224,224,3),9,np.uint8)]))
    np.testing.assert_array_equal(x.observed[1][1][0],np.zeros((5,2)))
    np.testing.assert_array_equal(x.observed[1][1][1],x.physical[0])
    d=x.decisions[1]
    assert d["observed"]["stack_sha256"]==p4.rgb_sha(x.observed[1][0])
    assert d["observed_context_used_for_selection"] is (arm=="actual")
    expected=d["observed"]["encoded_context"] if arm=="actual" else x.decisions[0]["model_next_context"]
    assert d["planning_context"]==expected
    assert d["planning_context_sha256"]==p4.content_sha(expected)


def test_imagined_selection_does_not_read_actual_terminal_feedback():
    a,b=exercise("imagined",terminal_value=9),exercise("imagined",terminal_value=200)
    assert a.decisions[1]["observed"]!=b.decisions[1]["observed"]
    assert a.decisions[1]["planning_context"]==b.decisions[1]["planning_context"]
    assert a.decisions[1]["selected"]==b.decisions[1]["selected"]
    actual=exercise("actual",terminal_value=200)
    assert actual.decisions[1]["planning_context"]["boundary"]==200
    assert actual.decisions[1]["planning_context"]!=b.decisions[1]["planning_context"]


@pytest.mark.parametrize("arm",p4.ARMS)
@pytest.mark.parametrize("abort_round",[0,1])
def test_guards_stop_without_padding_or_extra_future_selection(arm,abort_round):
    x=exercise(arm,abort_round=abort_round)
    assert x.result["status"]=="guarded_abort"
    assert x.result["physics_steps"]==5*abort_round+2
    assert len(x.observed)==len(x.decisions)==len(x.physical)==abort_round+1
    assert len(x.journal)==40*(abort_round+1)


@pytest.mark.parametrize("fault",["seal","replay"])
def test_predeployment_failure_refused(fault):
    with pytest.raises((OSError,ValueError)):
        exercise("actual",fail_seal=fault=="seal",bad_replay=fault=="replay")


def test_failed_candidate_retained_and_no_finite_subset_selection():
    events=[]
    count=0
    def replay(c,a,phase):
        nonlocal count
        count+=1
        if count==3:
            raise ValueError("injected")
        return {}
    with pytest.raises(RuntimeError,match="no deployable"):
        p4.select_pairs(context={},bank=bank(),paths=p4.paths_for(0)[:40],
            replay=replay,metric=lambda c:0.,journal=events.append)
    assert len(events)==40 and sum(e["completed"] for e in events)==39
    assert events[2]["error"]=="injected"


def test_ties_use_model_cost_then_lexical_path_only():
    chosen=p4.select_pairs(context={},bank=bank(),paths=[(9,3),(2,4)],
        replay=lambda c,a,p:{},metric=lambda c:1.,journal=lambda e:None)
    assert chosen["path"]==[2,4]


@pytest.mark.parametrize("index",[-1,4,True,1.5])
def test_no_new_root_indices(index):
    with pytest.raises(ValueError):
        p4.paths_for(index)


def test_fixed_seed_schedule_matches_v1_without_opening_outcomes():
    from experiments.spectral_reachability import p4_receding_runtime_20261027_v1 as old
    for i in range(4):
        assert p4.paths_for(i)==old.paths_for(i)
        assert len(set(p4.paths_for(i)))==80


@pytest.mark.parametrize("fault",["shape","nan","bounds","bool"])
def test_invalid_action_representation_refused(fault):
    b=bank()
    if fault=="shape":
        b=b[:15]
    elif fault=="nan":
        b[0,0,0]=np.nan
    elif fault=="bounds":
        b[0,0,0]=1.01
    else:
        b=b.astype(bool)
    with pytest.raises(ValueError):
        p4.validate_bank(b)


def test_cuda_failfast_before_any_model_access():
    with pytest.raises(ValueError,match="CPU neural fallback"):
        p4.encode_cuda(None,SimpleNamespace(type="cpu"),None,None,None,None)


def test_partial_packet_cannot_become_feedback():
    with pytest.raises(ValueError,match="partial/aborted"):
        p4.advance_observed(frames(),np.zeros((2,5,2)),frames()[0],bank()[0],
            {"status":"observation_unavailable","completed_commands":5,
             "remaining_commands_not_executed":0})


def rows_fixture():
    """Fake intercepted metadata; not saved CUDA receipts or physical results."""
    rows=[]
    for i in range(4):
        for arm in p4.ARMS:
            records=[]
            for iteration in range(2):
                for phase,horizons in (("planning",[2]*40),("evaluation",[2,1,1])):
                    records.extend({"round":iteration,"phase":phase,"method":arm,
                        "horizon_tokens":h,"predict_calls":h,"predict_samples":h,
                        "context_lengths":{3:h},"completed":True,"failed_predict_calls":0} for h in horizons)
            rows.append({"root_index":i,"arm":arm,"status":"two_rounds_completed",
                "rounds":[{"round":j,"execution":{"status":"completed","completed_commands":5,
                    "remaining_commands_not_executed":0}} for j in range(2)],
                "physics_steps":10,"predict_planning_calls":160,"predict_planning_samples":160,
                "predict_evaluation_calls":8,"predict_evaluation_samples":8,
                "image_encode_calls":2,"image_encode_frames":6,
                "action_encoder_work":{"calls":168,"samples":168},"predict":records})
    return rows


def test_report_engineering_only(tmp_path):
    r=p4.finalize(tmp_path,rows_fixture(),{},lambda:True,lambda:True,{}, {})
    assert all(r["gates"].values())
    assert r["status"]=="engineering_calibration_recorded"
    assert r["equal_round_budget_observed_all_pairs"]
    assert not any(r[k] for k in ("P1_admission","P4_admission","efficacy_verified","equal_total_compute"))
    assert r["scientific_status"]=="blocked_upstream"
    assert json.loads((tmp_path/"report.json").read_text())==json.loads(json.dumps(r))


@pytest.mark.parametrize("fault",["allocation","horizon","samples","method","failed",
    "missing","rounds","steps","action_work","image_work","totals","close"])
def test_forged_accounting_refused_even_if_branch_totals_match(tmp_path,fault):
    rows=rows_fixture()
    r=rows[0]
    resources={}
    if fault=="allocation":
        r["predict"][0]["round"]=1
    elif fault=="horizon":
        r["predict"][0]["horizon_tokens"]=1
    elif fault=="samples":
        r["predict"][0]["predict_samples"]=1
    elif fault=="method":
        r["predict"][0]["method"]="other"
    elif fault=="failed":
        r["predict"][0]["failed_predict_calls"]=1
    elif fault=="missing":
        r["predict"]=[]
    elif fault=="rounds":
        r["rounds"]=[]
    elif fault=="steps":
        r["physics_steps"]=9
    elif fault=="action_work":
        r["action_encoder_work"]["calls"]=167
    elif fault=="image_work":
        r["image_encode_frames"]=3
    elif fault=="totals":
        r["predict_evaluation_samples"]=7
    else:
        resources["setup_failure"]={"error":"close"}
    report=p4.finalize(tmp_path,rows,{},lambda:True,lambda:True,{},resources)
    assert report["status"]=="integrity_fail"


def test_guard_abort_kept_and_never_claims_all_pair_budget_matching(tmp_path):
    rows=rows_fixture()
    r=rows[0]
    r.update(status="guarded_abort",physics_steps=2,predict_planning_calls=80,
        predict_planning_samples=80,predict_evaluation_calls=4,predict_evaluation_samples=4,
        image_encode_calls=1,image_encode_frames=3,action_encoder_work={"calls":84,"samples":84})
    r["rounds"]=r["rounds"][:1]
    r["rounds"][0]["execution"].update(status="observation_unavailable",completed_commands=2,
        remaining_commands_not_executed=3)
    r["predict"]=r["predict"][:43]
    report=p4.finalize(tmp_path,rows,{},lambda:True,lambda:True,{}, {})
    assert report["status"]=="engineering_calibration_recorded"
    assert len(report["rows"])==8
    assert not report["equal_round_budget_observed_all_pairs"]
    assert not report["all_branches_complete"]


def test_source_and_input_failure_still_save_report(tmp_path):
    path=tmp_path/"input"
    path.write_text("not expected")
    def fail():
        raise ValueError("source mismatch")
    report=p4.finalize(tmp_path,[],{str(path):"wrong"},fail,lambda:True,{}, {})
    assert report["status"]=="integrity_fail"
    assert not report["gates"]["inputs_unchanged"]
    assert not report["gates"]["sources_unchanged"]
    assert report["failures"][0]["message"]=="source mismatch"


def test_exclusive_json_does_not_overwrite(tmp_path):
    path=tmp_path/"decision.json"
    p4.write(path,{"first":True})
    with pytest.raises(FileExistsError):
        p4.write(path,{"second":True})
    assert json.loads(path.read_text())=={"first":True}


@pytest.mark.parametrize("status,steps",[("completed",5),("observation_unavailable",0),
    ("observation_unavailable",2),("observation_unavailable",5),
    ("environment_terminated",3),("execution_failed",1)])
def test_all_guarded_trajectories_persist_before_evaluation(tmp_path,status,steps):
    packet=bank()[3]
    terminal=frames()[0]
    def fake_executor(env,commands,root_id,agent_rgb,on_observation):
        for i in range(steps):
            env.step(commands[i])
            on_observation(i+1,commands[i],(None,0.,False,False,{}),terminal)
        return {"status":status,"completed_commands":steps,"planned_commands":5,
            "remaining_commands_not_executed":5-steps,"terminal_rgb":terminal,
            "guard":{"non_neural_fake":True}}
    prefix=tmp_path/"round0"
    fake_env=SimpleNamespace(step=lambda command:(None,0.,False,False,{}))
    value,rgb=p4.execute_packet(env=fake_env,packet=packet,iteration=0,root_id="fixture",
        prefix=prefix,executor=fake_executor,state_of=lambda env:np.zeros(7))
    saved=json.loads((tmp_path/"round0.execution.json").read_text())
    assert saved==value and saved["status"]==status
    assert len(saved["trace"])==steps
    assert len(saved["attempt_before_step"])==steps
    assert len(list(tmp_path.glob("*.attempt_*.json")))==steps
    assert len(list(tmp_path.glob("*.step_*.json")))==steps
    np.testing.assert_array_equal(np.load(tmp_path/"round0.terminal.npy",allow_pickle=False),rgb)


def test_invalid_execution_receipt_is_saved_before_rejection(tmp_path):
    def fake_executor(*args,**kwargs):
        return {"status":"completed","completed_commands":5,"planned_commands":5,
                "terminal_rgb":frames()[0]}
    with pytest.raises(ValueError,match="trace differs"):
        p4.execute_packet(env=None,packet=bank()[0],iteration=0,root_id="fixture",
            prefix=tmp_path/"bad",executor=fake_executor,state_of=lambda env:np.zeros(7))
    assert (tmp_path/"bad.execution.json").exists()
    assert (tmp_path/"bad.terminal.npy").exists()


@pytest.mark.parametrize("fault",["continue_after_abort","negative","remaining","fake_abort"])
def test_invalid_round_physics_accounting_refused(fault):
    r=rows_fixture()[0]
    if fault=="continue_after_abort":
        r["rounds"][0]["execution"].update(status="observation_unavailable",completed_commands=2,
            remaining_commands_not_executed=3)
    elif fault=="negative":
        r["rounds"][1]["execution"]["completed_commands"]=-1
    elif fault=="remaining":
        r["rounds"][1]["execution"]["remaining_commands_not_executed"]=1
    else:
        r["status"]="guarded_abort"
    assert not p4.round_accounting_valid(r)


def test_actual_saved_bank_goal_q_untouched_offline_arithmetic_only():
    path=Path("results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json")
    if not path.exists():
        pytest.skip("saved calibration artifact absent")
    from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
    for row in json.loads(path.read_text())["rows"]:
        compiled=row["compiled"]
        b=[np.repeat(np.asarray(r["program_live_commands"][0])[None],5,axis=0)
           for r in compiled["receipts"]]
        np.testing.assert_array_equal(p4.validate_bank(b),b)
        goal=np.asarray(compiled["goal_vectors"])[0]
        assert metric_for(compiled)({"latent_history":[goal.tolist()]})==0.


def test_attempt_is_durable_before_environment_exception(tmp_path):
    prefix=tmp_path/"error"
    def fail_step(command):
        assert (tmp_path/"error.attempt_1.json").exists()
        raise RuntimeError("fake env.step failed")
    def fake_executor(env,commands,**kwargs):
        try:
            env.step(commands[0])
        except RuntimeError as exc:
            return {"status":"execution_failed","error":str(exc),"completed_commands":0,
                "attempted_commands":1,"planned_commands":5,"remaining_commands_not_executed":5,
                "terminal_rgb":frames()[0]}
        pytest.fail("expected failure")
    value,_=p4.execute_packet(env=SimpleNamespace(step=fail_step),packet=bank()[0],
        iteration=0,root_id="fixture",prefix=prefix,executor=fake_executor,
        state_of=lambda env:np.zeros(7))
    assert value["completed_commands"]==0 and len(value["trace"])==0
    assert len(value["attempt_before_step"])==1
    assert value["attempted_commands"]==1


def test_setup_failure_materializes_all_known_root_arms(tmp_path):
    rows=[]
    roots=["fixture_"+str(i) for i in range(4)]
    failure={"type":"RuntimeError","message":"non-neural setup failure"}
    p4.materialize_blocked(tmp_path,rows,roots,failure)
    assert len(rows)==8 and all(r["status"]=="blocked_setup_failure" for r in rows)
    report=p4.finalize(tmp_path,rows,{},lambda:True,lambda:True,{},{"setup_failure":failure})
    assert report["status"]=="integrity_fail"
    assert report["gates"]["eight_branches_retained"]
    assert not report["gates"]["no_runtime_errors"]
    assert len(list(tmp_path.glob("root_*.json")))==8


def test_setup_materialization_preserves_existing_executed_branch(tmp_path):
    rows=[{"root_index":0,"arm":"imagined","status":"runtime_error","marker":"preserve"}]
    original=deepcopy(rows[0])
    p4.materialize_blocked(tmp_path,rows,["fixture_"+str(i) for i in range(4)],{"error":"setup"})
    assert rows[0]==original and len(rows)==8
    assert not (tmp_path/"root_0.imagined.json").exists()
