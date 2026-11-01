"""CPU callback fixtures and durable journal bytes; no CUDA or physics."""
import json

import pytest

import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from test_p3_feedback_reverse_control_20261030_v1 import Session, setup
from experiments.spectral_reachability import p3_decision_rejection_capture_20261030_codex_v2 as m
from experiments.spectral_reachability import p3_decision_journal_20261030_codex_v1 as journal


@pytest.mark.parametrize("policy", m.POLICIES)
def test_rejection_is_durable_unchanged_and_fail_closed(tmp_path, monkeypatch, policy):
    root, raw = setup()
    original = dict(continue_expanding=True, program_type="R", extension_type="R", horizon=3,
                    additional={"original": "preserve"})
    monkeypatch.setattr(m.core, "choose", lambda *args: original)
    store = journal.Store(tmp_path / "journal", max_bytes=128 * 1024 * 1024,
                          terminal_reserve=32 * 1024 * 1024)
    class Sink:
        def __init__(self):
            self.events = []
        def event(self, value):
            self.events.append(value)
            store.event(value)
    sink = Sink()
    result = m.execute_slot(root, policy, raw, Session(),
                            lambda task, sink: fixture.RamBackend(task, sink), sink, permit_ram=True)
    records = [json.loads(line)["payload"] for line in store.path.read_bytes().splitlines()]
    rejected = [x for x in records if x.get("kind") == "decision_gate_rejected"]
    assert len(rejected) == 1 and rejected[0]["decision"] == original
    event = rejected[0]
    assert event["policy"] == policy and event["stage"] == 0
    assert event["root_sha256"] == m.core.digest(root)
    assert event["context_sha256"] == m.core.digest(root["root_context"])
    assert event["diagnostic"] and event["node"] and event["signals"] and len(event["responses"]) == 2
    assert event["signal_receipt"]["scheduling_signal"] == event["signals"]
    assert event["gate_error"] == "ValueError: unsupported decision"
    assert result["status"] == "failed" and result["error"] == event["gate_error"]
    final_receipt, gate_receipt = result["failed_decision_receipt"], event["resource_receipt"]
    assert {k: v for k, v in final_receipt.items() if k != "wall_ns"} == {
        k: v for k, v in gate_receipt.items() if k != "wall_ns"}
    assert final_receipt["wall_ns"] >= gate_receipt["wall_ns"]
    assert result["failed_decision_receipt"]["attempted"]["replay_calls"] == 0
    assert result["failed_decision_receipt"]["attempted"]["program_compile_calls"] == 0
    assert not result["packets"] and not any(x.get("kind") == "packet_attempt" for x in records)
    assert records[-2]["kind"] == "decision_gate_rejected" and records[-1]["kind"] == "slot_final"


def test_write_failure_does_not_dispatch(monkeypatch):
    root, raw = setup()
    monkeypatch.setattr(m.core, "choose", lambda *args: dict(
        continue_expanding=False, program_type="R", extension_type="R", horizon=1))
    class Sink:
        def __init__(self):
            self.events = []
        def event(self, value):
            if value.get("kind") == "decision_gate_rejected":
                raise OSError("write failure")
            self.events.append(value)
    sink = Sink()
    result = m.execute_slot(root, "signal", raw, Session(),
                            lambda task, store: fixture.RamBackend(task, store), sink, permit_ram=True)
    assert result["status"] == "failed" and result["error"] == "OSError: write failure"
    assert not result["packets"] and not any(e["kind"] == "packet_attempt" for e in sink.events)


@pytest.mark.parametrize("change", [
    {"continue_expanding": False}, {"program_type": "R"},
    {"extension_type": "R"}, {"horizon": 2}, {"horizon": True}])
def test_all_original_gate_rejections_are_preserved(monkeypatch, change):
    root, raw = setup()
    decision = dict(continue_expanding=True, program_type="progress", extension_type="progress", horizon=1)
    decision.update(change)
    monkeypatch.setattr(m.core, "choose", lambda *args: decision)
    sink = fixture.Events()
    result = m.execute_slot(root, "fixed_h1", raw, Session(),
                            lambda task, store: fixture.RamBackend(task, store), sink, permit_ram=True)
    rejected = [e for e in sink.events if e["kind"] == "decision_gate_rejected"]
    assert len(rejected) == 1 and rejected[0]["decision"] == decision
    assert result["status"] == "failed" and not result["packets"]


def test_admitted_path_matches_prior_behavior():
    root, raw = setup()
    def run(module):
        events = fixture.Events()
        return module.execute_slot(root, "signal", raw, Session(),
                                   lambda task, sink: fixture.RamBackend(task, sink),
                                   events, permit_ram=True), events.events
    new, events = run(m)
    previous, _ = run(m.prior)
    assert new["status"] == previous["status"] == "callback_complete_unverified"
    assert new["packets"] == previous["packets"]
    assert [x["selected_horizon"] for x in new["decisions"]] == [
        x["selected_horizon"] for x in previous["decisions"]]
    assert not any(e["kind"] == "decision_gate_rejected" for e in events)


def test_roster_fixed_when_gate_rejects(monkeypatch):
    root, raw = setup()
    monkeypatch.setattr(m.core, "choose", lambda *args: dict(
        continue_expanding=True, program_type="R", extension_type="R", horizon=3))
    sink = fixture.Events()
    closed = []
    roster = m.execute_roster([root, root],
        lambda slot, root_index, policy: (Session(), raw, lambda: closed.append(slot)),
        lambda task, store: fixture.RamBackend(task, store), sink, permit_ram=True)
    assert roster["fixed_denominator"] == 6 and closed == list(range(6))
    assert [r["status"] for r in roster["rows"]] == ["failed"] * 6
    assert len([e for e in sink.events if e["kind"] == "decision_gate_rejected"]) == 6
