import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_online_carry_cpu_codex_20261101 as m
from experiments.spectral_reachability import p1_carry_binding_20261101 as b


def test_dynamic_allocation_and_rejection():
    budget = b.extra_cost(2, 2, has_carry=True)
    n, cost = m.allocate(budget, budget, candidates=8, horizon=2, has_carry=True)
    assert n == 2 and cost == budget
    with pytest.raises(ValueError):
        m.allocate({k: 0 for k in budget}, budget, candidates=8, horizon=2, has_carry=True)


def test_dls_feedback_mix():
    bank = np.zeros((8, 3, 5, 2), dtype=np.float32)
    assert np.array_equal(m.dls_feedback_candidates(bank, 0), bank)
    assert np.any(m.dls_feedback_candidates(bank, 5)[4:])
    with pytest.raises(ValueError):
        m.dls_feedback_candidates(bank, 20)


def make_executor(tmp_path, *, issue=None, leaf=None, budget=None):
    def observe(boundary):
        return dict(rgb=np.full((2, 2, 3), boundary, np.uint8),
                    states=torch.zeros(1, 2, 2), history=torch.zeros(1, 1, 10),
                    goal=torch.ones(1, 2), mean=torch.zeros(2), std=torch.ones(2),
                    context_source="test_fixture")
    def candidates(boundary, obs):
        return torch.full((2, 3, 5, 2), .2), "fixture"
    cap = b.extra_cost(3, 3, has_carry=True)
    return m.OnlineExecutor(episode="e", leaf=leaf or m.DemoLeaf(), observe=observe,
        candidates=candidates, issue=issue or (lambda p: {"ok": True}),
        budget=budget or {k: v * 3 for k, v in cap.items()}, node_limit=cap,
        ledger=tmp_path / "ledger.jsonl")


def records(ex):
    return [json.loads(line) for line in ex.ledger.read_text().splitlines()]


def test_online_success_carry_provenance_and_settlement(tmp_path):
    ex = make_executor(tmp_path)
    ex.step()
    ex.step()
    events = records(ex)
    selections = [x for x in events if x["event"] == "selection"]
    assert selections[1]["selected_source"] == "carry"
    reservations = [x for x in events if x["event"] == "reservation"]
    assert reservations[1]["previous_plan_sha256"] == selections[0]["plan_sha256"]
    assert len([x for x in events if x["event"] == "issue_receipt"]) == 2
    for settlement in (x for x in events if x["event"] == "settlement"):
        assert settlement["actual"] == settlement["reserved"]
    assert ex.boundary == 10


def test_issue_failure_is_uncertain_and_not_retried(tmp_path):
    def fail(packet):
        raise RuntimeError("lost receipt")
    ex = make_executor(tmp_path, issue=fail)
    with pytest.raises(RuntimeError, match="lost receipt"):
        ex.step()
    events = records(ex)
    assert [x["event"] for x in events][-3:] == ["issue_intent", "failure", "settlement"]
    assert events[-1]["actual"]["predict_calls"] > 0
    with pytest.raises(RuntimeError, match="reconcile"):
        ex.step()


def test_partial_selector_failure_settles_attempted_calls(tmp_path):
    class Failing(m.DemoLeaf):
        def predict(self, states, actions, *, phase):
            raise RuntimeError("predict failed")
    ex = make_executor(tmp_path, leaf=Failing())
    with pytest.raises(RuntimeError, match="predict failed"):
        ex.step()
    events = records(ex)
    assert events[-1]["actual"]["predict_calls"] == 1
    assert events[-1]["actual"]["action_encode_calls"] == 1
    assert not any(x["event"] == "issue_intent" for x in events)
