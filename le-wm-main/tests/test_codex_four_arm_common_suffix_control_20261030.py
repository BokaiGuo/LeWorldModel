import copy

import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_common_suffix_control_20261030 as m
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


def task():
    return {"binding":{"root_sha256":"a"*64,"action_domain_sha256":"b"*64},
            "goal":[1.]*192,"projection":torch.eye(192).tolist(),
            "root_context":{"latent_history":torch.zeros(3,192).tolist(),
                            "action_history":torch.zeros(2,10).tolist()},
            "action_mean":[0.,0.],"action_std":[1.,1.]}


@pytest.mark.parametrize("arm", spec.ARMS)
def test_common_optimizer_and_all_initialization_charged(tmp_path, arm):
    t=task()
    policy,contract,config=m.contract_for(t,"c"*64,"d"*64)
    h=spec.digest(contract)
    driver=durable.Driver(tmp_path/"ledger",contract,h,arm,"node",arm)
    leaf=leafmod.LeafAdapter(Affine(),driver,contract,h,analytic_fixture=True)
    events=[]
    out=m.plan(leaf,arm,t,config,policy,contract,device="cpu",
               emit=lambda k,v:events.append((k,copy.deepcopy(v))))
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],h,arm,"node",arm)
    assert account["charged_upper_bounds"] == m.cost(arm)
    assert out["initial_raw"].shape == out["optimized_raw"].shape == (8,3,5,2)
    assert out["selected_raw"].shape == (3,5,2)
    packet=m.first_packet(out)
    assert packet.shape == (5,2)
    assert torch.equal(packet,out["optimized_raw"][out["selected_index"],0])
    assert len(out["scores"]) == 8
    assert [k for k,_ in events][-2:] == ["optimized_candidates","final_selection"]
    assert contract["identity"]["optimizer"] == m.spectral.file_sha(m.suffix.__file__)
    assert not out["fairness_proven"]


def test_event_failure_before_candidates_prevents_suffix(tmp_path):
    t=task()
    policy,contract,config=m.contract_for(t,"c"*64,"d"*64)
    h=spec.digest(contract)
    driver=durable.Driver(tmp_path/"ledger",contract,h,"random","node","random")
    leaf=leafmod.LeafAdapter(Affine(),driver,contract,h,analytic_fixture=True)
    def fail(kind,payload):
        if kind=="initial_candidates":
            raise OSError("sink failed")
    with pytest.raises(OSError,match="sink failed"):
        m.plan(leaf,"random",t,config,policy,contract,device="cpu",emit=fail)
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],h,"random","node","random")
    assert account["charged_upper_bounds"]["predict_calls"]==0
    assert account["status"]=="failed_or_blocked"


def test_same_caps_cover_all_arms_without_calling_them_equal_usage():
    _,c,_=m.contract_for(task(),"c"*64,"d"*64)
    caps=c["levels"][0]["caps"]
    assert all(all(m.cost(a)[k]<=caps[k] for k in spec.AXES) for a in spec.ARMS)
    assert m.cost("structured") != m.cost("random")
    assert m.cost("immediate_progress")["selection_evaluations"] > m.cost("random")["selection_evaluations"]


def test_first_packet_refuses_bad_shape_or_unbound_selection():
    optimized=torch.zeros((8,3,5,2))
    result={"optimized_raw":optimized,"selected_index":0,"selected_raw":optimized[0].clone()}
    assert m.first_packet(result).shape==(5,2)
    result["selected_raw"]=torch.zeros((5,2))
    with pytest.raises(ValueError):
        m.first_packet(result)
    result["selected_raw"]=torch.ones((3,5,2))
    with pytest.raises(ValueError):
        m.first_packet(result)
