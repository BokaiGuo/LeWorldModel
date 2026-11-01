from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import p1_common_budget_cem_events_20261101_leibniz as m
from experiments.spectral_reachability import codex_four_arm_common_suffix_control_20261030 as structured


class Analytic:
    def encode(self, info):
        return info
    def action_encoder(self, actions):
        return actions
    def predict(self, states, actions):
        return states + actions[:, -1, :].mean(dim=-1, keepdim=True)[:, None, :]


@pytest.fixture
def artifacts():
    root = Path(__file__).resolve().parents[1]
    task = json.loads((root / "results/p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json").read_bytes())["source"]["tasks"][0]
    _, template, _ = structured.contract_for(task, "a"*64, "b"*64)
    templates = {"structured": {b: deepcopy(template) for b in m.variable.BOUNDARIES},
                 "variable_cem": {b: m.variable.node_contract(task, "a"*64, "b"*64, b, seed=20261101)
                                  for b in m.variable.BOUNDARIES}}
    paths = m.required_sources() | {str(Path(x.__file__).resolve())
                                    for x in (m.spec, m.common.durable, m.common.leafmod)}
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths}
    base_protocol = m.common.make_protocol(templates, sources)
    return templates, m.protocol(base_protocol)


def controller(tmp_path, artifacts, boundary=0, stop=True):
    templates, protocol = artifacts
    contract = m.bind(protocol, m.spec.digest(protocol), "variable_cem", boundary,
                      templates["variable_cem"][boundary])
    sha = m.spec.digest(contract)
    driver = m.common.durable.Driver(tmp_path / "ledger", contract, sha, "native", "node", "fixture")
    leaf = m.common.leafmod.LeafAdapter(Analytic(), driver, contract, sha, analytic_fixture=True)
    goal, matrix = torch.ones(1, 2), torch.eye(2)
    mean, std = torch.zeros(2), torch.ones(2)
    identity = dict(model_sha256="a"*64, root_sha256="b"*64, task_sha256="c"*64,
                    goal_sha256=m.base.tensor_sha(goal), metric_sha256=m.base.tensor_sha(matrix),
                    normalizer_sha256=m.base.tensor_sha(torch.stack((mean, std))))
    def observe(boundary, metered):
        return dict(identity=identity, rgb_sha256="d"*64, provenance="fixture", encoding_mode="inherited",
                    states=torch.zeros(1, 3, 2), history=torch.zeros(1, 2, 10),
                    goal=goal, matrix=matrix, mean=mean, std=std)
    packets = []
    def issue(packet):
        packets.append(packet)
        return {"executed_steps": 1 if stop else 5, "stopped": stop}
    c = m.Controller(leaf=leaf, observe=observe, issue=issue, journal=tmp_path / "cem.jsonl",
                     run_id="fixture", identity=identity, seed=20261101,
                     budget_protocol=protocol, budget_sha=m.spec.digest(protocol))
    c.index = m.variable.BOUNDARIES.index(boundary)
    return c, driver, packets


@pytest.mark.parametrize("boundary,forward,events", [(0, 195, 65), (5, 194, 97),
                                                    (10, 193, 193), (15, 193, 193)])
def test_real_cem_controller_and_real_driver(tmp_path, artifacts, boundary, forward, events):
    c, driver, packets = controller(tmp_path, artifacts, boundary)
    result = c.step()
    c.close()
    receipt = driver.finish()
    assert result["boundary"] == boundary and len(packets) == 1
    assert driver.charged["predict_calls"] == driver.charged["action_encode_calls"] == forward
    assert driver.charged["replay_calls"] == driver.charged["selection_evaluations"] == events
    assert receipt["status"] == "closed_unverified"
    rows = [json.loads(line) for line in driver.path.read_bytes().splitlines()]
    intents = [r["payload"]["kind"] for r in rows if r["kind"] == "intent"]
    assert intents.count("replay") == intents.count("selection") == events
    assert result["receipt"] == {"executed_steps": 1, "stopped": True}
    with pytest.raises((RuntimeError, ValueError)):
        c.step()


def test_common_caps_bind_both_arms_and_old_cap_insufficient(artifacts):
    templates, protocol = artifacts
    for arm in m.common.ARMS:
        for b in m.variable.BOUNDARIES:
            assert m.bind(protocol, m.spec.digest(protocol), arm, b, templates[arm][b])["levels"][0]["caps"] == m.CAPS
    assert m.common.CAPS["replay_calls"] == 64
    assert all((4 * n + 1) > 64 for n in (16, 24, 48, 48))


def test_wrong_node_and_protocol_rejected_before_issue(tmp_path, artifacts):
    c, driver, packets = controller(tmp_path, artifacts, 0)
    c.index = 1
    with pytest.raises(ValueError, match="identity drift"):
        c.step()
    assert packets == [] and all(v == 0 for v in driver.charged.values())
    c.close()
    driver.finish()
    _, protocol = artifacts
    digest = m.spec.digest(protocol)
    protocol["caps"]["selection_evaluations"] = 194
    with pytest.raises(ValueError):
        m.bind(protocol, digest, "structured", 0, artifacts[0]["structured"][0])


def test_model_failure_preserves_attempt_and_never_issues(tmp_path, artifacts):
    c, driver, packets = controller(tmp_path, artifacts)
    def fail(*args):
        raise RuntimeError("fixture model failure")
    c.leaf.model.predict = fail
    with pytest.raises(RuntimeError, match="fixture model failure"):
        c.step()
    assert c.stopped and not packets and driver.failed
    c.close()
    assert driver.finish()["status"] == "failed_or_blocked"
    rows = [json.loads(line) for line in driver.path.read_bytes().splitlines()]
    assert any(row["kind"] == "failure" for row in rows)
    assert driver.charged["replay_calls"] == 1
    assert driver.charged["selection_evaluations"] == 0


def test_fixed_seed_instrumented_matches_original_controller(tmp_path, artifacts):
    left = tmp_path / "instrumented"
    right = tmp_path / "original"
    left.mkdir()
    right.mkdir()
    metered, driver_a, _ = controller(left, artifacts)
    helper, driver_b, _ = controller(right, artifacts)
    original = m.variable.Controller(leaf=helper.leaf, observe=helper.observe, issue=helper.issue,
                                     journal=right / "original.jsonl", run_id="original",
                                     identity=helper.identity, seed=helper.seed)
    actual = metered.step()
    reference = original.step()
    assert actual["selected_index"] == reference["selected_index"]
    assert (actual["selected_raw"] == reference["selected_raw"]).all()
    metered.close()
    helper.close()
    original.close()
    driver_a.finish()
    driver_b.finish()
