import copy

import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_own_planners_20261030 as m
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


def setup(tmp_path, arm):
    task = {"binding":{"root_sha256":"a"*64,"action_domain_sha256":"b"*64},
            "goal":[1.]*192,"projection":torch.eye(192).tolist(),
            "root_context":{"latent_history":torch.zeros(3,192).tolist(),
                            "action_history":torch.zeros(2,10).tolist()},
            "action_mean":[0.,0.],"action_std":[1.,1.]}
    policy, contract, config = m.contract_for(task, arm, "c"*64, "d"*64)
    sha = spec.digest(contract)
    driver = durable.Driver(tmp_path/"ledger",contract,sha,arm,"node",arm)
    leaf=leafmod.LeafAdapter(Affine(),driver,contract,sha,analytic_fixture=True)
    return task, policy, contract, config, sha, driver, leaf


@pytest.mark.parametrize("arm", spec.ARMS)
def test_actual_per_arm_planning_and_charged_final_replay(tmp_path, arm):
    task,policy,contract,cfg,sha,driver,leaf = setup(tmp_path, arm)
    events=[]
    result=m.plan(leaf,arm,task,cfg,policy,contract,device="cpu",
                  emit=lambda k,v:events.append((k,copy.deepcopy(v))))
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],sha,arm,"node",arm)
    assert account["charged_upper_bounds"] == m.cost(arm)
    assert result["selected_raw"].shape==(3,5,2)
    assert len(result["scores"]) == 8
    assert events[-1][0] == "selected"
    assert not result["fairness_proven"]
    if arm == "native":
        assert "not_installed_native" in result["label"]


def test_feedback_task_change_rejected_before_prediction(tmp_path):
    task,policy,contract,cfg,sha,driver,leaf = setup(tmp_path,"random")
    task["root_context"]["latent_history"][2][0]=1.
    with pytest.raises(ValueError,match="binding"):
        m.plan(leaf,"random",task,cfg,policy,contract,device="cpu",emit=lambda *_:None)
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],sha,"random","node","random")
    assert account["charged_upper_bounds"]["predict_calls"]==0


def test_persistence_failure_stops_before_model(tmp_path):
    task,policy,contract,cfg,sha,driver,leaf=setup(tmp_path,"native")
    def emit(kind,payload):
        if kind == "candidates":
            raise OSError("write failure")
    with pytest.raises(OSError,match="write failure"):
        m.plan(leaf,"native",task,cfg,policy,contract,device="cpu",emit=emit)
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],sha,"native","node","native")
    assert account["charged_upper_bounds"]["predict_calls"]==0
    assert account["status"]=="failed_or_blocked"
