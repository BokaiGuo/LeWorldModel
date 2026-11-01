import copy

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_four_arm_prefix_intervention_20261030 as m


def test_fixed_four_by_three_by_two_actions_are_paired():
    trajectories={arm:np.full((20,2),i/10,dtype=np.float32).tolist()
                  for i,arm in enumerate(m.ARMS)}
    greedy=np.full((5,2),-.2,np.float32).tolist()
    rows=m.make_schedule(trajectories,greedy)
    assert len(rows)==24
    for arm in m.ARMS:
        arm_rows=[r for r in rows if r["trajectory_arm"]==arm]
        assert [(r["prefix_role"],r["suffix_role"]) for r in arm_rows]==[
            (p,s) for p in m.PREFIXES for s in m.SUFFIXES]
        for r in arm_rows:
            expected=trajectories[arm][5:15] if r["suffix_role"]=="saved" else [[0.0,0.0]]*10
            assert r["suffix"]==expected
            if r["prefix_role"]=="selected":
                assert r["prefix"]==trajectories[arm][:5]
            if r["prefix_role"]=="greedy":
                assert r["prefix"]==greedy


def test_schedule_rejects_bad_actions():
    t={arm:np.zeros((20,2),np.float32).tolist() for arm in m.ARMS}
    t["native"][0][0]=1.1
    with pytest.raises(ValueError):
        m.make_schedule(t,np.zeros((5,2),np.float32).tolist())


def test_censor_requires_matching_return_and_proven_finite_out_of_bounds():
    branch={"index":2,"commands":np.zeros((15,2),np.float32).tolist()}
    state=[1.,513.,3.,4.,.2,0.,0.]
    event={"branch":2,"step":6,"command":[0.,0.],"state":state}
    receipt={"phase":"plan","step":6,"state":state,"snapshot":{"state":state},
             "pixel_sha256":"a"*64,"file_sha256":"b"*64}
    proof=m.validate_censor(branch,event,receipt)
    assert proof["out_of_bounds"]==[{"coordinate":1,"value":513.}]
    for mut in (dict(event,branch=3),dict(event,step=7),dict(event,command=[1.,0.]),
                dict(event,state=[1.,2.,3.,4.,.2,0.,0.])):
        with pytest.raises(ValueError):
            m.validate_censor(branch,mut,receipt)
    bad=copy.deepcopy(receipt)
    bad["snapshot"]["state"][1]=512
    with pytest.raises(ValueError):
        m.validate_censor(branch,event,bad)


def test_cli_partial_is_nonzero(monkeypatch,capsys):
    monkeypatch.setattr(m,"run",lambda *args:{"status":"partial_or_failed"})
    monkeypatch.setattr("sys.argv",["intervention","run","--prepared","x","--sha256","x",
                                    "--output","x","--authorization","x"])
    assert m.main()==1


@pytest.mark.parametrize("out_of_bounds,expected",[(True,"censored_out_of_domain"),(False,"raise")])
def test_real_guard_censor_only_for_returned_out_of_domain(tmp_path,monkeypatch,out_of_bounds,expected):
    state=np.array([10.,513. if out_of_bounds else 100.,20.,30.,.2,0.,0.],np.float64)
    goal=np.zeros(7)
    goal_rgb=np.zeros((2,2,3),np.uint8)
    class Body:
        latest_action=np.zeros(2,np.float32)
        def _get_obs(self):
            return state.copy()
    class Env:
        unwrapped=Body()
        def step(self,action):
            self.unwrapped.latest_action=action.copy()
            return {"state":state.copy()},0.,False,False,{"goal_state":goal,"goal":goal_rgb,"n_contacts":3}
    class Store:
        path=tmp_path
        def emit(self,kind,**kwargs):
            pass
    def capture(env,obs,row,store,phase,step):
        pixels=np.zeros((224,224,3),np.uint8)
        path=tmp_path/"actual.png"
        Image.fromarray(pixels).save(path)
        row["last_observation"]={"phase":phase,"step":step,"state":state.tolist(),
            "snapshot":{"state":state.tolist()},"path":str(path),"file_sha256":m.sha(path),
            "pixel_sha256":m.old.array_sha(pixels)}
        m.old.kernel.moving.guard(pixels,obs["state"])
    monkeypatch.setattr(m.moving,"setup",lambda *args:True)
    monkeypatch.setattr(m.old,"capture",capture)
    monkeypatch.setattr(m.old,"native_transition",lambda *args:None)
    branch={"index":0,"commands":np.zeros((15,2),np.float32).tolist()}
    row={"control_returned":0,"executed_raw":[],"transitions":[]}
    if expected=="raise":
        with pytest.raises(ValueError,match="no proven canvas violation"):
            m.episode(Env(),{"goal":goal,"goal_rgb":goal_rgb},{"plan":None},branch,row,
                      type("Meter",(),{"region":""})(),Store())
    else:
        m.episode(Env(),{"goal":goal,"goal_rgb":goal_rgb},{"plan":None},branch,row,
                  type("Meter",(),{"region":""})(),Store())
        assert row["status"]==expected and row["censor"]["step"]==1 and row["endpoint"] is None
        assert row["native_success"] is None and row["final_metrics"] is None
        assert row["transitions"][0]["n_contacts"]==3


def test_censored_continues_other_branch_but_unrelated_failure_stops_and_preserves_denominator(tmp_path,monkeypatch):
    branches=m.make_schedule({a:np.zeros((20,2),np.float32).tolist() for a in m.ARMS},
                             np.zeros((5,2),np.float32).tolist())
    monkeypatch.setattr(m,"verify",lambda *args:{"schedule":branches,"bindings":{}})
    monkeypatch.setattr(m.source_launcher,"verify",lambda *args:{"source":{}})
    monkeypatch.setattr(m.p3,"source_data",lambda *args:{"references":[{}]})
    monkeypatch.setattr(m,"exclusive",lambda path,kind:tmp_path)
    monkeypatch.setattr(m,"sha",lambda path:"a"*64)
    class Store:
        instance=None
        def __init__(self,path):
            Store.instance=self
            self.path=path
            self.previous="tail"
            self.saved={}
        def put(self,name,value,**kwargs):
            self.saved[name]=copy.deepcopy(value)
        def emit(self,kind,**kwargs):
            pass
        def check(self):
            pass
    class Meter:
        def __init__(self,*args):
            self.counts={k:{"attempted":0,"completed":0} for k in m.CAPS}
            self.pending={}
            self.secondary=[]
        def call(self,kind,fn):
            return fn()
    class Env:
        def __init__(self,meter):
            self.unwrapped=type("Body",(),dict(_live_meter=meter,_live_constructor_complete=True,
                with_target=False,render_action=False,render_size=224,relative=True,
                action_scale=100,control_hz=10,dt=.01))()
        def close(self):
            pass
    monkeypatch.setattr(m,"Store",Store)
    monkeypatch.setattr(m,"Meter",Meter)
    monkeypatch.setattr(m.old,"native_factory",Env)
    called=[]
    def episode(env,ref,data,branch,row,meter,store):
        called.append(branch["index"])
        if branch["index"]==0:
            row["status"]="censored_out_of_domain"
        else:
            raise RuntimeError("unexpected error")
    monkeypatch.setattr(m,"episode",episode)
    result=m.run("p","a"*64,str(tmp_path),m.AUTH)
    report=Store.instance.saved["report.json"]
    assert called==[0,1] and result["status"]=="partial_or_failed"
    assert len(report["rows"])==24 and report["rows"][0]["status"]=="censored_out_of_domain"
    assert report["rows"][1]["status"]=="failed"
    assert all(r["status"]=="not_started" for r in report["rows"][2:])
