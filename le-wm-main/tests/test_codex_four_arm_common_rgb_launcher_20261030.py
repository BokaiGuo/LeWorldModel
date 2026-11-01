from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_four_arm_common_rgb_launcher_20261030 as m
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from test_codex_four_arm_common_suffix_control_20261030 import task
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


class Store:
    def __init__(self, path):
        self.path=path
        self.finalizing=False
        self.previous="tail"
        self.saved={}
    def put(self,name,value,**kwargs):
        self.saved[name]=deepcopy(value)
    def emit(self,kind,**kwargs):
        pass
    def namespace(self,name):
        p=self.path/name
        p.mkdir()
        return p
    def check(self):
        pass


class Meter:
    def __init__(self,store,check):
        self.counts={"physical":{"attempted":0,"completed":0}}
        self.pending=[]
        self.secondary=[]
    def call(self,kind,fn):
        return fn()


def test_execute_connects_common_node_all_arms_and_keeps_fixed_roster(tmp_path,monkeypatch):
    monkeypatch.setattr(m.p3,"Meter",Meter)
    monkeypatch.setattr(m,"sha",lambda path:"a"*64)
    seen=[]
    class Env:
        def __init__(self,meter):
            self.unwrapped=type("Body",(),dict(_live_meter=meter,_live_constructor_complete=True,
                with_target=False,render_action=False,render_size=224,relative=True,
                action_scale=100,control_hz=10,dt=.01))()
        def close(self):
            pass
    def episode(env,ref,data,layer,base,row,nodes,visual,runtime,meter,store,plan):
        seen.append((row["arm"],plan))
        row.update(status="completed20",control_returned=20,full_horizon=True)
    monkeypatch.setattr(m.rgb,"episode",episode)
    prep={"source":{"tasks":[{}],"layers":[{}]}}
    data={"references":[{}]}
    report=m.execute(prep,data,Store(tmp_path),None,{"check_limits":lambda:None},Env)
    assert [x[0] for x in seen]==list(m.spec.ARMS)
    assert all(fn is m.bridge.plan_node for _,fn in seen)
    assert report["status"]=="recorded_known_root_not_admitted"
    assert len(report["nodes"])==4
    assert all(len(nodes)==4 for nodes in report["nodes"].values())


def test_failed_arm_blocks_later_arms_and_partial_cli_is_nonzero(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m.p3,"Meter",Meter)
    monkeypatch.setattr(m,"sha",lambda path:"a"*64)
    def fail(*args):
        raise RuntimeError("guard failed")
    monkeypatch.setattr(m.rgb,"episode",fail)
    class Env:
        def __init__(self,meter):
            self.unwrapped=type("Body",(),dict(_live_meter=meter,_live_constructor_complete=True,
                with_target=False,render_action=False,render_size=224,relative=True,
                action_scale=100,control_hz=10,dt=.01))()
        def close(self):
            pass
    report=m.execute({"source":{"tasks":[{}],"layers":[{}]}},{"references":[{}]},
                     Store(tmp_path),None,{"check_limits":lambda:None},Env)
    assert report["status"]=="partial_or_failed"
    assert report["rows"][0]["first_error"] is not None
    assert all(r["status"]=="not_started" for r in report["rows"][1:])
    monkeypatch.setattr(m,"run",lambda *args:{"status":"partial_or_failed"})
    monkeypatch.setattr("sys.argv",["launcher","run","--prepared","x","--sha256","x",
                                    "--output","x","--authorization","x"])
    assert m.main()==1


def test_prepare_binds_independent_bridge_and_runtime_sources(monkeypatch):
    monkeypatch.setattr(m.p3,"verify",lambda *args:{"sources":{}})
    value=m.build()
    paths=value["sources"]
    for module in (m, m.bridge, m.control, m.rgb, m.control.spectral, m.control.suffix,
                   m.control.raw, m.control.replay, m.control.leafmod, m.control.durable):
        assert str(Path(module.__file__).resolve()) in paths
    assert not value["protocol"]["fairness_proven"]
    assert value["protocol"]["node_costs"]["structured"] != value["protocol"]["node_costs"]["random"]


def test_verify_rejects_changed_sources_and_tampered_prepare(tmp_path,monkeypatch):
    results=tmp_path/"results"
    results.mkdir()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    path=results/(m.NAME+"_fixture.json")
    fixed={"schema":"fixture","sources":{"source":"old"}}
    path.write_bytes(m.raw(fixed))
    expected=m.sha(path)
    monkeypatch.setattr(m,"build",lambda:fixed)
    assert m.verify(str(path),expected)==fixed
    monkeypatch.setattr(m,"build",lambda:{"schema":"fixture","sources":{"source":"new"}})
    with pytest.raises(ValueError,match="drift"):
        m.verify(str(path),expected)
    path.write_bytes(m.raw({"schema":"tampered"}))
    with pytest.raises(ValueError,match="SHA/path"):
        m.verify(str(path),expected)


def test_full_launcher_loop_uses_real_node_bridge_with_cpu_substitutes(tmp_path,monkeypatch):
    monkeypatch.setattr(m.p3,"Meter",Meter)
    monkeypatch.setattr(m,"sha",lambda path:"a"*64)
    original=leafmod.LeafAdapter.__init__
    monkeypatch.setattr(leafmod.LeafAdapter,"__init__",
        lambda self,model,driver,contract,contract_sha:
        original(self,model,driver,contract,contract_sha,analytic_fixture=True))
    monkeypatch.setattr(m.old,"state_seal",lambda body:"same")
    def setup(env,ref,plan,row,meter,store):
        row["last_observation"]={"arm":row["arm"],"step":0}
        return True
    monkeypatch.setattr(m.rgb.moving,"setup",setup)
    feedback=[]
    def encode(visual,runtime,task,layer,row,obs,body,meter,store,boundary):
        assert obs=={"arm":row["arm"],"step":boundary}
        feedback.append((row["arm"],boundary))
        return np.full(192,boundary,np.float32),{"boundary":boundary}
    monkeypatch.setattr(m.rgb.moving,"encode_boundary",encode)
    def roll(context,z,raw10,base):
        result=deepcopy(context)
        result["latent_history"][-1]=z.tolist()
        return result
    monkeypatch.setattr(m.rgb.moving,"roll",roll)
    monkeypatch.setattr(m.p3,"local_task",
        lambda base,context,identity:dict(base,root_context=deepcopy(context)))
    def packet(env,commands,*,start,row,ref,meter,store):
        assert np.asarray(commands).shape==(5,2)
        row["executed_raw"].extend(commands)
        row["control_returned"]+=5
        return {"arm":row["arm"],"step":start+5}
    monkeypatch.setattr(m.rgb.moving,"packet",packet)
    class Env:
        def __init__(self,meter):
            self.unwrapped=type("Body",(),dict(_live_meter=meter,_live_constructor_complete=True,
                with_target=False,render_action=False,render_size=224,relative=True,
                action_scale=100,control_hz=10,dt=.01))()
        def close(self):
            pass
    runtime={"model":Affine(),"model_sha256":"c"*64,"hardware_sha256":"d"*64,
             "device":"cpu","check_limits":lambda:None}
    store=Store(tmp_path)
    data={"references":[{"raw_history":np.zeros((10,2),np.float32)}],
          "plan":None,"visual_tasks":{0:None}}
    report=m.execute({"source":{"tasks":[task()],"layers":[None]}},data,store,None,runtime,Env)
    assert report["status"]=="recorded_known_root_not_admitted"
    assert feedback==[(arm,b) for arm in m.spec.ARMS for b in (0,5,10,15)]
    assert all(row["control_returned"]==20 for row in report["rows"])
    assert all(node["status"]=="sealed" and node["receipt"]["account"]["status"]=="closed_unverified"
               for nodes in report["nodes"].values() for node in nodes)
