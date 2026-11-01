"""Offline non-neural fixture tests for real progress/R branch plumbing."""
from copy import deepcopy
import runpy

import numpy as np
import pytest

from experiments.spectral_reachability import p3_decision_dual_branch_20261030_codex_v1 as m
from test_p3_feedback_reverse_control_20261030_v1 import Session, setup

helpers = runpy.run_path(str(m.core.ROOT / "tests/test_agent_p3_r_native_orchestrator_20261029_v1.py"))


def original(branch):
    return dict(continue_expanding=True, program_type=branch, extension_type=branch,
                candidates=4 if branch == "progress" else 8, horizon=3)


@pytest.mark.parametrize("branch", ["progress", "reconfiguration"])
def test_real_groups_exact_branch_budget_and_unchanged_decision(monkeypatch, branch):
    task = helpers["r_task"]()
    store = helpers["Memory"]()
    decision = original(branch)
    before = deepcopy(decision)
    monkeypatch.setattr(m.core, "choose", lambda *args: decision)
    result = m.execute_decision(task, task["root_context"], 0, "signal",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert decision == before == result["decision"]
    assert result["branch"] == branch
    assert result["status"] == "dual_panels_complete_not_certified", result["error"]
    assert result["selected_horizon"] == 3 and len(result["selected_actions"]) == 5
    receipt = result["resource_receipt"]
    assert receipt["caps"] == receipt["attempted"] == receipt["completed"] == m.decision_cost(branch)
    assert len(result["panels"]) == 2
    assert all(len(group["rows"]) == (4 if branch == "progress" else 8) for group in result["panels"])
    assert [e["payload"]["branch"] for e in store.events
            if e.get("kind") == "io" and e["payload"].get("kind") == "branch_budget"] == [branch]


def test_static_budgets_and_type_gate():
    p, r = m.decision_cost("progress"), m.decision_cost("reconfiguration")
    assert (p["predict_API"], p["VJP_directions"], p["replay_calls"], p["program_compile_calls"]) == (32, 384, 12, 8)
    assert (r["predict_API"], r["VJP_directions"], r["replay_calls"], r["program_compile_calls"]) == (106, 3840, 40, 20)
    assert p != r
    for policy, horizon in (("fixed_h1", 1), ("signal", 3), ("reverse_signal", 1)):
        assert m.choose(policy, original("progress")) == horizon
        assert m.choose(policy, original("reconfiguration")) == horizon
    for bad in (dict(original("progress"), candidates=8), dict(original("reconfiguration"), candidates=4),
                dict(original("progress"), program_type="composition"),
                dict(original("progress"), continue_expanding=False)):
        with pytest.raises(ValueError):
            m.validate_decision(bad)


def test_no_witness_no_packet_selection(monkeypatch):
    task = helpers["r_task"]()
    task["goal"] = [0.] * 191 + [1.]
    task["binding"]["goal_sha256"] = m.core.digest(task["goal"])
    store = helpers["Memory"]()
    monkeypatch.setattr(m.core, "choose", lambda *args: original("reconfiguration"))
    result = m.execute_decision(task, task["root_context"], 0, "fixed_h1",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "failed" and result["selected_actions"] is None
    assert [g["status"] for g in result["panels"]] == ["no_witness", "no_witness"]
    assert result["resource_receipt"]["attempted"]["replay_calls"] == 36


def test_feedback_replans_after_actual_encoding_and_failure_stops_packet(monkeypatch):
    task, raw = setup()
    seen = []
    def decide(root, context, stage, *args, **kwargs):
        seen.append((stage, deepcopy(context)))
        return dict(status="dual_panels_complete_not_certified" if stage == 0 else "failed",
                    error="fixture refusal" if stage else None, selected_actions=[[0., 0.]] * 5)
    monkeypatch.setattr(m, "execute_decision", decide)
    store = helpers["Memory"]()
    result = m.execute_slot(task, "signal", raw, Session(), None, store, lambda *args: None,
                            kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "failed" and len(result["packets"]) == 1
    assert result["callback_attempted"] == result["callback_completed"] == {"packet": 1, "encode": 1}
    assert [x[0] for x in seen] == [0, 1] and seen[0][1] != seen[1][1]


@pytest.mark.parametrize("branch", ["progress", "reconfiguration"])
def test_actual_branch_can_drive_a_fixture_packet(monkeypatch, branch):
    task = helpers["r_task"]()
    store = helpers["Memory"]()
    monkeypatch.setattr(m.core, "choose", lambda *args: original(branch))
    class Feedback:
        episode_id = "fixture-episode"
        def packet(self, start, raw):
            assert start == 0
            return dict(raw=raw, rgb_bytes=b"", end=5, stopped=True, episode_id=self.episode_id)
        def encode(self, *args):
            raise AssertionError("stopped packet must not encode")
    result = m.execute_slot(task, "signal", np.zeros((5, 2), np.float32), Feedback(),
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda stage, cid, call: store.capsule(stage, cid, call),
        kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "native_stopped", result["error"]
    assert result["decisions"][0]["branch"] == branch and len(result["packets"]) == 1
    assert result["callback_completed"] == {"packet": 1, "encode": 0}


def test_progress_partial_failure_keeps_attempted_row_and_no_action(monkeypatch):
    task = helpers["r_task"]()
    store = helpers["Memory"]()
    monkeypatch.setattr(m.core, "choose", lambda *args: original("progress"))
    def fail(*args):
        raise RuntimeError("synthetic generator failure")
    monkeypatch.setattr(m.r.live, "generate", fail)
    result = m.execute_decision(task, task["root_context"], 0, "signal",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "failed" and result["selected_actions"] is None
    assert result["panels"][0]["rows"][0]["status"] == "attempted"
    assert result["resource_receipt"]["attempted"]["program_compile_calls"] == 1
    assert result["resource_receipt"]["completed"]["program_compile_calls"] == 0
