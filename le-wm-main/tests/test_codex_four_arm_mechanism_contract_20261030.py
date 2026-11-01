import copy
import json

import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_mechanism_contract_20261030 as m
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


def setup(tmp_path, arm):
    task = {"binding": {"root_sha256": "a"*64, "action_domain_sha256": "b"*64},
            "goal": [1.]*192, "projection": torch.eye(192).tolist(),
            "root_context": {"latent_history": torch.zeros(3,192).tolist(),
                             "action_history": torch.zeros(2,10).tolist()},
            "action_mean": [0.,0.], "action_std": [1.,1.]}
    cfg = {"steps":1,"learning_rate":.001,"history_size":3}
    policy, contract = m.make_contract(task, cfg, "c"*64, "d"*64)
    sha = m.spec.digest(contract)
    driver = durable.Driver(tmp_path/"ledger", contract, sha, arm, "common", arm)
    leaf = leafmod.LeafAdapter(Affine(), driver, contract, sha, analytic_fixture=True)
    return task, cfg, policy, contract, sha, driver, leaf


@pytest.mark.parametrize("arm", ["structured", "random", "immediate_progress", "native"])
def test_all_arms_paid_shared_suffix_and_final_replay(tmp_path, arm):
    task, cfg, policy, contract, sha, driver, leaf = setup(tmp_path, arm)
    events = []
    result = m.run_mechanism(leaf, arm, task, cfg, policy, contract, device="cpu",
                             emit=lambda k,v: events.append((k,copy.deepcopy(v))))
    receipt = driver.finish()
    account = consumer.consume(receipt["journal"], receipt["sha256"], sha, arm, "common", arm)
    assert account["charged_upper_bounds"] == m.expected_cost(arm, cfg)
    assert result["optimized_raw"].shape == (8,3,5,2)
    assert len(result["scores"]) == 8
    assert not result["fairness_proven"]
    if arm == "immediate_progress":
        assert sum(kind == "screening_score" for kind,_ in events) == 16
    if arm == "native":
        assert result["label"] == "native_style_initialization_variant_not_installed_native_cem"


def test_same_seed_determinism_and_native_first_mean():
    x = m.initial_samples("native")
    assert torch.equal(x, m.initial_samples("native"))
    assert x.shape == (8,3,5,2) and torch.count_nonzero(x[0]) == 0
    assert x.abs().max() <= 1
    assert m.initial_samples("immediate_progress").shape[0] == 16


def test_task_or_policy_drift_rejected_before_model(tmp_path):
    task,cfg,policy,contract,sha,driver,leaf = setup(tmp_path, "random")
    task["goal"][0] = 2
    with pytest.raises(ValueError, match="binding"):
        m.run_mechanism(leaf, "random", task, cfg, policy, contract, device="cpu", emit=lambda *_:None)
    account = consumer.consume((receipt:=driver.finish())["journal"], receipt["sha256"], sha, "random", "common", "random")
    assert account["charged_upper_bounds"]["predict_calls"] == 0


def test_screening_failure_does_not_enter_suffix(tmp_path, monkeypatch):
    task,cfg,policy,contract,sha,driver,leaf = setup(tmp_path, "immediate_progress")
    def fail(*args):
        raise RuntimeError("predict failure")
    monkeypatch.setattr(leaf.model, "predict", fail)
    with pytest.raises(RuntimeError, match="predict failure"):
        m.run_mechanism(leaf, "immediate_progress", task, cfg, policy, contract, device="cpu", emit=lambda *_:None)
    receipt=driver.finish()
    account=consumer.consume(receipt["journal"],receipt["sha256"],sha,"immediate_progress","common","immediate_progress")
    assert account["status"] == "failed_or_blocked"
    assert account["charged_upper_bounds"]["predict_calls"] == 1
    assert account["charged_upper_bounds"]["backward_calls"] == 0


def test_performance_layer_not_runnable_by_mechanism_adapter():
    info = m.performance_requirements()
    assert info["implementation_ready"] is False
    assert info["shared_suffix_is_performance_evidence"] is False
    assert set(info["required_own_optimizers"]) == {"structured","native","random","immediate_progress"}


def test_performance_draft_requires_common_feedback_and_keeps_unequal_costs():
    zeros = dict.fromkeys(m.spec.AXES, 0)
    phases = ("setup", "initialization", "search", "final_replay", "physical")
    arms = {}
    for i, arm in enumerate(m.spec.ARMS):
        costs = {p: dict(zeros) for p in phases}
        costs["search"]["predict_calls"] = i+1
        arms[arm] = {"feedback_sha256":"f"*64, "optimizer_source_sha256":str(i+1)*64,
                     "initializer_source_sha256":"a"*64, "phase_costs":costs}
    draft = {"schema":m.PERF_SCHEMA, "feedback_sha256":"f"*64, "arms":arms,
             "caps":{axis:10 for axis in m.spec.AXES}}
    result = m.check_performance_draft(draft)
    assert [result["declared_totals"][a]["predict_calls"] for a in m.spec.ARMS] == [1,2,3,4]
    assert result["implementation_ready"] is False and result["fairness_proven"] is False
    bad = copy.deepcopy(draft)
    bad["arms"]["native"]["feedback_sha256"] = "e"*64
    with pytest.raises(ValueError, match="feedback"):
        m.check_performance_draft(bad)
    bad = copy.deepcopy(draft)
    bad["caps"]["predict_calls"] = 3
    with pytest.raises(ValueError, match="cap"):
        m.check_performance_draft(bad)
    bad = copy.deepcopy(draft)
    bad["arms"]["random"]["observed_score"] = 0
    with pytest.raises(ValueError, match="fields"):
        m.check_performance_draft(bad)
