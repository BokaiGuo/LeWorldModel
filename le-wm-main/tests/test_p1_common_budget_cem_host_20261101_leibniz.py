import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import p1_common_budget_cem_host_20261101_leibniz as m


class Analytic:
    def encode(self, info): return info
    def action_encoder(self, actions): return actions
    def predict(self, states, actions):
        return states + actions[:, -1, :].mean(dim=-1, keepdim=True)[:, None, :]


def make_node(tmp_path):
    root = Path(__file__).resolve().parents[1]
    task = json.loads((root / "results/p1_first_feedback_carry_20261101_main_root0_PREP_MAIN02.json").read_bytes())["source"]["tasks"][0]
    contract = m.node_contract(task, "a"*64, "b"*64, 0)
    driver = m.old.durable.Driver(tmp_path / "ledger", contract, m.spec.digest(contract), "native", "node", "s0_full_cem")
    leaf = m.old.leafmod.LeafAdapter(Analytic(), driver, contract, m.spec.digest(contract), analytic_fixture=True)
    live = m.LiveLeaf({"check_limits": lambda: None}, lambda: None)
    live.node = leaf
    return live, driver


def test_common_budget_exact_costs_and_cap():
    for b, n in zip((0, 5, 10, 15), (65, 97, 193, 193)):
        assert m.node_cost(b)["replay_calls"] == m.node_cost(b)["selection_evaluations"] == n
        assert m.node_cost(b)["predict_calls"] <= m.budget.CAPS["predict_calls"]
    assert m.make_protocol({}, 0)["node_caps_by_boundary"]["0"] == m.budget.CAPS


def test_live_leaf_real_controller_cost_and_winner_receipt(tmp_path):
    live, driver = make_node(tmp_path)
    goal, matrix = torch.ones(1, 2), torch.eye(2)
    mean, std = torch.zeros(2), torch.ones(2)
    identity = dict(model_sha256="a"*64, root_sha256="b"*64, task_sha256="c"*64,
                    goal_sha256=m.old.cem.tensor_sha(goal), metric_sha256=m.old.cem.tensor_sha(matrix),
                    normalizer_sha256=m.old.cem.tensor_sha(torch.stack((mean, std))))
    def observe(boundary, metered):
        return dict(identity=identity, rgb_sha256="d"*64, provenance="fixture", encoding_mode="inherited",
                    states=torch.zeros(1, 3, 2), history=torch.zeros(1, 2, 10),
                    goal=goal, mean=mean, std=std, matrix=matrix)
    dispatched = []
    def issue(packet):
        m.previous.verify_node_events(c.journal.path, 0)
        dispatched.append(packet)
        return {"executed_steps": 1, "stopped": True}
    c = m.Controller(leaf=live, observe=observe, issue=issue, journal=tmp_path/"cem.jsonl",
                     run_id="test", identity=identity, seed=m.old.SEED)
    c.step()
    c.close()
    assert len(dispatched) == 1 and c.stopped
    assert driver.charged == m.node_cost(0)
    assert driver.finish()["status"] == "closed_unverified"
    rows = [json.loads(x) for x in driver.path.read_bytes().splitlines()]
    assert sum(x["kind"] == "intent" and x["payload"]["kind"] == "selection" for x in rows) == 65


def test_missing_active_node_is_fail_closed(tmp_path):
    live = m.LiveLeaf({"check_limits": lambda: None}, lambda: None)
    with pytest.raises(ValueError, match="actual RGB node"):
        _ = live.driver
