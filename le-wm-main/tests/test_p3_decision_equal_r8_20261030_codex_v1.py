"""R matched-control tests with explicit RAM native stand-ins; never CUDA/physics."""
from copy import deepcopy
import runpy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_decision_equal_r8_20261030_codex_v1 as m
from test_p3_feedback_reverse_control_20261030_v1 import Session, setup

helpers = runpy.run_path(str(m.core.ROOT / "tests/test_agent_p3_r_native_orchestrator_20261029_v1.py"))


def test_budget_accounts_both_full_R_groups_and_verification():
    cost = m.decision_cost()
    assert cost["predict_API"] == 106
    assert cost["VJP_directions"] == 3840
    assert cost["jacobian_API"] == 20
    assert cost["replay_calls"] == 40
    assert cost["program_compile_calls"] == 20
    assert cost["array_SVD_calls"] == 22
    assert cost["solve_calls"] == 18 and cost["metric_calls"] == 36


@pytest.mark.parametrize("policy,expected", [("fixed_h1", 1), ("signal", 3), ("reverse_signal", 1)])
def test_real_R_routing_preserves_original(policy, expected):
    decision = dict(continue_expanding=True, program_type="reconfiguration",
                    extension_type="reconfiguration", horizon=3, candidates=8)
    before = deepcopy(decision)
    assert m.choose(policy, decision) == expected and decision == before
    with pytest.raises(ValueError):
        m.choose(policy, dict(decision, program_type="progress"))


def test_feedback_replans_only_from_returned_rgb(monkeypatch):
    root, raw = setup()
    states, caps = [], []
    def decision(root, context, stage, policy, factory, sink, capsule, **kwargs):
        states.append(deepcopy(context))
        caps.append(capsule)
        return dict(status="R_panels_complete_not_certified", error=None,
                    selected_actions=[[0.0, 0.0]] * 5)
    monkeypatch.setattr(m, "execute_decision", decision)
    sink = helpers["Memory"]()
    result = m.execute_slot(root, "signal", raw, Session(), None, sink, lambda *a: None,
                            kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "callback_complete_not_certified"
    assert len(states) == 4 and len(result["packets"]) == 4
    assert result["callback_completed"] == {"packet": 4, "encode": 4}
    phases = [e["kind"] for e in sink.events if e["kind"] in
              ("packet_attempt", "packet_return", "encode_attempt", "encode_return")]
    assert phases == ["packet_attempt", "packet_return", "encode_attempt", "encode_return"] * 4


def test_failed_decision_prevents_packet(monkeypatch):
    root, raw = setup()
    monkeypatch.setattr(m, "execute_decision", lambda *a, **kw:
                        dict(status="failed", error="no witness", selected_actions=None))
    sink = helpers["Memory"]()
    result = m.execute_slot(root, "signal", raw, Session(), None, sink, lambda *a: None,
                            kind="RAM_FIXTURE", permit_ram=True)
    assert result["status"] == "failed" and result["packets"] == []
    assert result["callback_attempted"] == {"packet": 0, "encode": 0}


def test_roster_keeps_six_attempts_and_cleanup(monkeypatch):
    root, raw = setup()
    monkeypatch.setattr(m, "execute_decision", lambda *a, **kw:
                        dict(status="failed", error="no witness", selected_actions=None))
    sink, closed = helpers["Memory"](), []
    def sessions(slot, root_index, policy):
        if slot == 2:
            raise RuntimeError("setup failure")
        return Session(), raw, lambda: closed.append(slot)
    result = m.execute_roster([root, root], sessions, None, sink, lambda *a: None,
                              kind="RAM_FIXTURE", permit_ram=True)
    assert result["fixed_denominator"] == 6 and len(result["rows"]) == 6
    assert [r["status"] for r in result["rows"]] == [
        "failed", "failed", "setup_failed", "failed", "failed", "failed"]
    assert closed == [0, 1, 3, 4, 5]


def test_host_factory_routes_real_native_tree_sink_and_latches_failure(monkeypatch):
    from experiments.spectral_reachability import p3_decision_native_cuda_adapter_20261030_codex_v1 as a
    root, _ = setup()
    adapter = object.__new__(a.Adapter)
    adapter.poisoned = adapter.closed = False
    adapter.active_slot = 0
    adapter.prepared = {"tasks": [root, root]}
    adapter.model, adapter.device = object(), object()
    adapter.factory = SimpleNamespace(poisoned=False)
    calls = []
    adapter.guard = lambda label: calls.append(label)
    class Tree:
        kind = "HOST_CUDA"
        def __init__(self, model, device, task, sink):
            assert model is adapter.model and device is adapter.device and callable(sink.event)
        def signal(self, *args):
            raise RuntimeError("native failure")
    monkeypatch.setattr(a.identity.host.prior, "NativeTree", Tree)
    factory = m.host_backend_factory(adapter)
    task = m.r.parent.parent_task(root, root["root_context"])
    backend = factory(task, helpers["Memory"]())
    assert backend.kind == "HOST_CUDA" and calls == ["R_backend_construct"]
    with pytest.raises(RuntimeError, match="native failure"):
        backend.signal({}, object())
    assert adapter.poisoned and adapter.factory.poisoned
    with pytest.raises(ValueError):
        factory(task, helpers["Memory"]())


def test_real_r_evaluator_is_called_after_original_r_gate(monkeypatch):
    task = helpers["r_task"]()
    store = helpers["Memory"]()
    original = dict(continue_expanding=True, program_type="reconfiguration",
                    extension_type="reconfiguration", horizon=3, candidates=8)
    monkeypatch.setattr(m.core, "choose", lambda *a: original)
    seen = []
    def capture(root, node, decision, mapping, session, reserve, emit, on_start):
        seen.append((deepcopy(decision), deepcopy(mapping["native_response"]["horizon"])))
        raise RuntimeError("fixture stop before R dispatch")
    monkeypatch.setattr(m.r, "evaluate_R", capture)
    result = m.execute_decision(task, task["root_context"], 0, "signal",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert result["decision"] == original
    assert result["status"] == "failed" and "fixture stop" in result["error"]
    assert seen == [(dict(extension_type="reconfiguration", candidates=8, horizon=1), 1)]
    assert result["selected_actions"] is None


def test_real_dual_r_group_ram_fixture(monkeypatch):
    task = helpers["r_task"]()
    store = helpers["Memory"]()
    decision = dict(continue_expanding=True, program_type="reconfiguration",
                    extension_type="reconfiguration", horizon=3, candidates=8)
    monkeypatch.setattr(m.core, "choose", lambda *a: decision)
    result = m.execute_decision(task, task["root_context"], 0, "signal",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert result["decision"] == decision
    assert len(result["panels"]) == 2, result["error"]
    assert [g["bank"]["horizon"] for g in result["panels"]] == [1, 3]
    assert all(len(g["rows"]) == 8 and g["control"]["status"] == "complete" for g in result["panels"])
    assert [g["status"] for g in result["panels"]] == ["witness_found", "witness_found"]
    assert result["status"] == "R_panels_complete_not_certified", result["error"]
    assert result["resource_receipt"]["attempted"] == m.decision_cost()
    assert result["selected_horizon"] == 3 and len(result["selected_actions"]) == 5


def test_real_dual_r_no_witness_never_falls_back(monkeypatch):
    task = helpers["r_task"]()
    task["goal"] = [0.] * 191 + [1.]
    task["binding"]["goal_sha256"] = m.core.digest(task["goal"])
    store = helpers["Memory"]()
    monkeypatch.setattr(m.core, "choose", lambda *a: dict(
        continue_expanding=True, program_type="reconfiguration", extension_type="reconfiguration",
        horizon=3, candidates=8))
    result = m.execute_decision(task, task["root_context"], 0, "fixed_h1",
        lambda t, s: helpers["NonlinearRAM"](t, s), store,
        lambda cid, call: store.capsule(0, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    assert [g["status"] for g in result["panels"]] == ["no_witness", "no_witness"]
    assert result["status"] == "failed" and result["selected_actions"] is None
    assert result["resource_receipt"]["attempted"]["replay_calls"] == 36
