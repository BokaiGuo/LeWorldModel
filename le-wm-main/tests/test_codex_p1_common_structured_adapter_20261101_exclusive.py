from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import codex_p1_common_structured_adapter_20261101_exclusive as m
from experiments.spectral_reachability import codex_four_arm_consumer_20261030 as consumer
from test_codex_four_arm_common_suffix_control_20261030 import task
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


def setup(tmp_path):
    t = task()
    policy, template, config = m.original.contract_for(t, "c" * 64, "d" * 64)
    modules = (m, *m.DEPENDENCIES, m.common, m.common.spec, m.common.durable, m.common.leafmod)
    paths = {Path(x.__file__).resolve() for x in modules}
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    templates = {arm: {b: deepcopy(template) for b in m.common.BOUNDARIES} for arm in m.common.ARMS}
    protocol = m.common.make_protocol(templates, sources)
    driver, leaf, contract = m.common.open_node(tmp_path / "common", protocol, m.spec.digest(protocol),
                                                "structured", 0, template, Affine(), analytic_fixture=True)
    return t, policy, template, config, protocol, driver, leaf, contract


def test_real_structured_planner_matches_legacy_and_meter(tmp_path):
    t, policy, template, config, protocol, driver, leaf, contract = setup(tmp_path)
    records = []
    result = m.plan(leaf, t, protocol, m.spec.digest(protocol), 0, device="cpu",
                    emit=lambda kind, payload: records.append(kind))
    receipt = driver.finish()
    account = consumer.consume(receipt["journal"], receipt["sha256"], m.spec.digest(contract),
                               "structured", "node", "structured_b00")
    assert account["charged_upper_bounds"] == m.original.cost("structured")
    assert records[-1] == "final_selection"
    assert result["physical_execution_authorized"] is False
    old = m.common.durable.Driver(tmp_path / "old", template, m.spec.digest(template), "structured", "node", "old")
    old_leaf = m.leafmod.LeafAdapter(Affine(), old, template, m.spec.digest(template), analytic_fixture=True)
    expected = m.original.plan(old_leaf, "structured", t, config, policy, template,
                               device="cpu", emit=lambda *_: None)
    old.finish()
    assert torch.equal(result["optimized_raw"], expected["optimized_raw"])
    assert result["scores"] == expected["scores"]
    assert result["selected_index"] == expected["selected_index"]


def test_budget_preflight_aborts_before_planner_calls(tmp_path):
    t, _, _, _, protocol, driver, leaf, _ = setup(tmp_path)
    vector = dict.fromkeys(m.spec.AXES, 0)
    vector["predict_calls"] = 195
    vector["predict_samples"] = 195
    driver.dispatch("verification", "predict", vector, lambda: None)
    with pytest.raises(ValueError, match="full node reserve unavailable"):
        m.plan(leaf, t, protocol, m.spec.digest(protocol), 0, device="cpu", emit=lambda *_: None)
    assert driver.failed
    assert driver.charged["predict_calls"] == 195
    assert driver.finish()["status"] == "failed_or_blocked"


def test_drift_and_unbound_sources_fail_stop(tmp_path):
    t, _, _, _, protocol, driver, leaf, _ = setup(tmp_path)
    t["goal"][0] += 1
    with pytest.raises(ValueError):
        m.plan(leaf, t, protocol, m.spec.digest(protocol), 0, device="cpu", emit=lambda *_: None)
    assert driver.failed
    driver.finish()


def test_emit_failure_is_fail_stop(tmp_path):
    t, _, _, _, protocol, driver, leaf, _ = setup(tmp_path)
    def fail(*_):
        raise OSError("sink")
    with pytest.raises(OSError, match="sink"):
        m.plan(leaf, t, protocol, m.spec.digest(protocol), 0, device="cpu", emit=fail)
    assert driver.failed
    assert driver.charged["predict_calls"] == 0
    driver.finish()


def test_wrong_boundary_rejected_before_calls(tmp_path):
    t, _, _, _, protocol, driver, leaf, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="contract binding"):
        m.plan(leaf, t, protocol, m.spec.digest(protocol), 5, device="cpu", emit=lambda *_: None)
    assert driver.failed and driver.charged["predict_calls"] == 0
    driver.finish()


def test_v2_common_budget_real_planner(tmp_path):
    t, _, template, _, protocol, driver, _, _ = setup(tmp_path)
    driver.finish()
    for name in m.common_v2.required_sources():
        protocol["sources"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    v2 = m.common_v2.protocol(protocol)
    contract = m.common_v2.bind(v2, m.spec.digest(v2), "structured", 0, template)
    assert contract["levels"][0]["caps"]["replay_calls"] == 193
    driver = m.common.durable.Driver(tmp_path / "v2", contract, m.spec.digest(contract),
                                     "structured", "node", "structured_b00")
    leaf = m.leafmod.LeafAdapter(Affine(), driver, contract, m.spec.digest(contract), analytic_fixture=True)
    result = m.plan(leaf, t, v2, m.spec.digest(v2), 0, device="cpu", emit=lambda *_: None)
    assert result["selected_raw"].shape == (3, 5, 2)
    assert driver.charged == m.original.cost("structured")
    driver.finish()
