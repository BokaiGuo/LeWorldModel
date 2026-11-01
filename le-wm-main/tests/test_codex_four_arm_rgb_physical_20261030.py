from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_rgb_physical_20261030 as m


class Store:
    def __init__(self):
        self.events = []
    def put(self, name, value):
        self.events.append((name, deepcopy(value)))
    def emit(self, kind, **value):
        self.events.append((kind, deepcopy(value)))


class Env:
    unwrapped = object()


def test_episode_replans_from_each_own_actual_observation(monkeypatch):
    monkeypatch.setattr(m.old,"state_seal",lambda body:"unchanged")
    base={"binding":{},"root_context":{"layout":"states_with_between_state_actions_v1","context_length":3,"latent_dim":192,"action_dim":10,"latent_history":torch.zeros(3,192).tolist(),"action_history":torch.zeros(2,10).tolist()},
          "goal":[0.]*192,"action_mean":[0.,0.],"action_std":[1.,1.]}
    ref={"raw_history":np.zeros((10,2),np.float32)}
    row={"episode_id":"s0_random","arm":"random","executed_raw":[],"control_returned":0,
         "completed_packets":0,"last_observation":{"id":0},"encodings":[]}
    monkeypatch.setattr(m.moving,"setup",lambda *args:True)
    def encode(visual,runtime,task,layer,row,obs,body,meter,store,boundary):
        assert runtime == {"model":"visual"}
        assert obs["id"] == boundary
        return np.full(192,boundary,np.float32), {"boundary":boundary}
    monkeypatch.setattr(m.moving,"encode_boundary",encode)
    monkeypatch.setattr(m.p3,"local_task",lambda base,context,identity: dict(base,root_context=deepcopy(context)))
    seen=[]
    def node(task,arm,boundary,store,runtime):
        seen.append((boundary,task["root_context"]["latent_history"][-1][0],
                     deepcopy(task["root_context"]["action_history"])))
        return {"commands":np.full((5,2),.1,np.float32).tolist()}
    def packet(env,commands,*,start,row,ref,meter,store):
        row["executed_raw"].extend(commands);row["control_returned"]+=5
        return {"id":start+5}
    monkeypatch.setattr(m.moving,"packet",packet)
    meter=type("Meter",(),{"region":""})()
    nodes=[{"boundary":b,"status":"not_started"} for b in m.BOUNDARIES]
    m.episode(Env(),ref,{"plan":None,"visual_tasks":{0:None}},None,base,row,nodes,None,
              {"visual_runtime":{"model":"visual"},"other":"planner-only"},meter,Store(),node)
    assert [(a,b) for a,b,_ in seen] == [(0,0.),(5,5.),(10,10.),(15,15.)]
    assert row["control_returned"]==20 and row["status"]=="completed20"
    assert all(n["status"]=="sealed" for n in nodes)
    assert seen[1][2][-1] == pytest.approx([.1]*10)


def test_failed_node_prevents_any_packet(monkeypatch):
    monkeypatch.setattr(m.old,"state_seal",lambda body:"unchanged")
    row={"episode_id":"s0_random","arm":"random","executed_raw":[],"control_returned":0,
         "completed_packets":0,"last_observation":{"id":0},"encodings":[]}
    base={"root_context":{"latent_history":torch.zeros(3,192).tolist(),"action_history":torch.zeros(2,10).tolist()},
          "action_mean":[0.,0.],"action_std":[1.,1.]}
    monkeypatch.setattr(m.moving,"setup",lambda *args:True)
    monkeypatch.setattr(m.moving,"encode_boundary",lambda *args:(np.zeros(192,np.float32),{"boundary":0}))
    monkeypatch.setattr(m.p3,"local_task",lambda base,context,identity:dict(base,root_context=context))
    monkeypatch.setattr(m.moving,"packet",lambda *args,**kwargs:pytest.fail("must not step"))
    def fail(*args):
        raise RuntimeError("planner failed")
    nodes=[{"boundary":b,"status":"not_started"} for b in m.BOUNDARIES]
    with pytest.raises(RuntimeError,match="planner failed"):
        m.episode(Env(),{"raw_history":np.zeros((10,2),np.float32)},{"plan":None,"visual_tasks":{0:None}},
                  None,base,row,nodes,None,{},type("Meter",(),{"region":""})(),Store(),fail)
    assert row["control_returned"]==0 and nodes[0]["status"]=="attempted"


def test_partial_cli_is_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(m,"run",lambda *args:{"status":"partial_or_failed"})
    monkeypatch.setattr("sys.argv",["launcher","run","--prepared","x","--sha256","x",
                                    "--output","x","--authorization","x"])
    assert m.main()==1
    assert "partial_or_failed" in capsys.readouterr().out


def test_native_stop_on_twentieth_return_records_horizon(monkeypatch):
    monkeypatch.setattr(m.old,"state_seal",lambda body:"same")
    monkeypatch.setattr(m.moving,"setup",lambda *args:True)
    monkeypatch.setattr(m.moving,"encode_boundary",lambda *args:(np.zeros(192,np.float32),{}))
    monkeypatch.setattr(m.moving,"roll",lambda context,*args:context)
    monkeypatch.setattr(m.p3,"local_task",lambda base,context,identity:base)
    def packet(env,commands,*,start,row,ref,meter,store):
        row["executed_raw"].extend(commands)
        row["control_returned"]+=5
        if start==15:
            row["status"]="native_stopped_control"
            return None
        return {}
    monkeypatch.setattr(m.moving,"packet",packet)
    row={"episode_id":"s0_random","arm":"random","executed_raw":[],"control_returned":0,
         "completed_packets":0,"last_observation":{}}
    base={"root_context":{"latent_history":torch.zeros(3,192).tolist()},
          "action_mean":[0.,0.],"action_std":[1.,1.]}
    nodes=[{"boundary":b,"status":"not_started"} for b in m.BOUNDARIES]
    m.episode(Env(),{"raw_history":np.zeros((10,2),np.float32)},{"plan":None,"visual_tasks":{0:None}},
              None,base,row,nodes,None,{},type("Meter",(),{"region":""})(),Store(),
              lambda *args:{"commands":np.zeros((5,2)).tolist()})
    assert row["full_horizon"] and row["completed_packets"]==4
    assert row["status"]=="native_stopped_control"
