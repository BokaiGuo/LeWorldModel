import torch
import pytest
import numpy as np
from copy import deepcopy

from experiments.spectral_reachability import codex_four_arm_common_suffix_bridge_20261030 as bridge
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_rgb_physical_20261030 as physical
from test_codex_four_arm_common_suffix_control_20261030 import task
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


class Store:
    def __init__(self,path):
        self.path=path
        self.saved={}
    def namespace(self,name):
        p=self.path/name
        p.mkdir()
        return p
    def check(self):
        pass
    def put(self,name,value):
        self.saved[name]=value
    def emit(self,kind,**value):
        pass


def test_real_node_bridge_extracts_first_packet_and_consumes_ledger(tmp_path,monkeypatch):
    original=leafmod.LeafAdapter.__init__
    def fixture_init(self,model,driver,contract,contract_sha):
        original(self,model,driver,contract,contract_sha,analytic_fixture=True)
    monkeypatch.setattr(leafmod.LeafAdapter,"__init__",fixture_init)
    store=Store(tmp_path)
    runtime={"model":Affine(),"model_sha256":"c"*64,"hardware_sha256":"d"*64,
             "device":"cpu","check_limits":lambda:None}
    result=bridge.plan_node(task(),"random",0,store,runtime)
    assert torch.tensor(result["commands"]).shape==(5,2)
    assert result["account"]["status"]=="closed_unverified"
    assert result["account"]["charged_upper_bounds"]==bridge.control.cost("random")
    assert len(store.saved)==1
    assert (tmp_path/"s0_random_b00_common"/"candidates.jsonl").is_file()


def test_limit_failure_never_yields_commands(tmp_path,monkeypatch):
    original=leafmod.LeafAdapter.__init__
    monkeypatch.setattr(leafmod.LeafAdapter,"__init__",
        lambda self,model,driver,contract,contract_sha:
        original(self,model,driver,contract,contract_sha,analytic_fixture=True))
    def blocked():
        raise RuntimeError("cap")
    runtime={"model":Affine(),"model_sha256":"c"*64,"hardware_sha256":"d"*64,
             "device":"cpu","check_limits":blocked}
    with pytest.raises(ValueError,match="node failed"):
        bridge.plan_node(task(),"random",0,Store(tmp_path),runtime)


def test_existing_episode_seam_passes_each_arm_own_feedback(tmp_path,monkeypatch):
    original=leafmod.LeafAdapter.__init__
    monkeypatch.setattr(leafmod.LeafAdapter,"__init__",
        lambda self,model,driver,contract,contract_sha:
        original(self,model,driver,contract,contract_sha,analytic_fixture=True))
    monkeypatch.setattr(physical.old,"state_seal",lambda body:"same")
    monkeypatch.setattr(physical.moving,"setup",lambda *args:True)
    monkeypatch.setattr(physical.moving,"encode_boundary",
        lambda visual,runtime,task,layer,row,obs,body,meter,store,boundary:
        (np.full(192,obs["arm_value"]+boundary,np.float32),{"boundary":boundary}))
    def roll(context,z,raw10,base):
        out=deepcopy(context)
        out["latent_history"][-1]=z.tolist()
        return out
    monkeypatch.setattr(physical.moving,"roll",roll)
    monkeypatch.setattr(physical.p3,"local_task",
        lambda base,context,identity:dict(base,root_context=deepcopy(context)))
    def packet(env,commands,*,start,row,ref,meter,store):
        assert np.asarray(commands).shape==(5,2)
        row["executed_raw"].extend(commands)
        row["control_returned"]+=5
        return {"arm_value":env.arm_value}
    monkeypatch.setattr(physical.moving,"packet",packet)
    runtime={"model":Affine(),"model_sha256":"c"*64,"hardware_sha256":"d"*64,
             "device":"cpu","check_limits":lambda:None}
    store=Store(tmp_path)
    seen=[]
    def node(t,arm,boundary,store,runtime):
        seen.append((arm,boundary,t["root_context"]["latent_history"][-1][0]))
        return bridge.plan_node(t,arm,boundary,store,runtime)
    for arm,value in (("native",1),("random",9)):
        env=type("Env",(),{"arm_value":value,"unwrapped":object()})()
        row={"episode_id":"s0_"+arm,"arm":arm,"executed_raw":[],"control_returned":0,
             "completed_packets":0,"last_observation":{"arm_value":value}}
        nodes=[{"boundary":b,"status":"not_started"} for b in physical.BOUNDARIES]
        physical.episode(env,{"raw_history":np.zeros((10,2),np.float32)},
                         {"plan":None,"visual_tasks":{0:None}},None,task(),row,nodes,None,
                         runtime,type("Meter",(),{"region":""})(),store,node)
        assert row["control_returned"]==20
    assert seen==[(arm,b,value+b) for arm,value in (("native",1),("random",9))
                  for b in physical.BOUNDARIES]
