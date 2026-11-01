import json
from pathlib import Path

import pytest

import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from test_p3_feedback_reverse_control_20261030_v1 import Session, setup
from experiments.spectral_reachability import p3_decision_journal_20261030_codex_v1 as writer
from experiments.spectral_reachability import p3_decision_saved_consumer_20261030_codex_v1 as reader


class Stop(Session):
    def packet(self, start, raw):
        return dict(raw=raw[:2].copy(), rgb_bytes=b"", end=start + 2, stopped=True, episode_id=self.episode_id)


class Failed(Session):
    def encode(self, end, rgb):
        raise RuntimeError("intentional fixture failure")


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    directory = tmp_path_factory.mktemp("p3_decision") / "run"
    store = writer.Store(directory, max_bytes=128 * 1024 * 1024, terminal_reserve=32 * 1024 * 1024)
    class FixtureSink:
        def __init__(self):
            self.events = []
        def event(self, event):
            store.event(event)
            self.events.append(event)
    sink = FixtureSink()
    root, prior = setup()
    active = {"slot": None}
    def sessions(slot, root_index, policy):
        active["slot"] = slot
        if slot == 3:
            raise RuntimeError("setup")
        session = Session() if slot == 0 else Failed() if slot == 2 else Stop()
        return session, prior.copy(), lambda: None
    class BrokenBackend(fixture.RamBackend):
        def replay(self, *args):
            raise RuntimeError("replay")
    def backend(task, sink):
        return (BrokenBackend if active["slot"] == 5 else fixture.RamBackend)(task, sink)
    result = writer.control.execute_roster([root, root], sessions,
        backend, sink, permit_ram=True)
    digest = store.finish(result)
    return directory / "journal.jsonl", digest


def rewrite(path, payloads):
    previous = None
    lines = []
    for i, payload in enumerate(payloads):
        body = dict(index=i, previous=previous, payload=payload)
        record = dict(body, sha256=writer.control.core.digest(body))
        lines.append(writer.control.core.canonical(record) + b"\n")
        previous = record["sha256"]
    raw = b"".join(lines)
    path.write_bytes(raw)
    return writer.sha(raw)


def test_real_durable_roundtrip_with_failure_and_stop(saved):
    path, digest = saved
    report = reader.consume(path, digest, 128 * 1024 * 1024)
    assert report["slot_status"] == ["callback_complete_unverified", "native_stopped", "failed",
                                     "setup_failed", "native_stopped", "failed"]
    assert len(report["slot_accounts"][0]["decision_accounts"]) == 4
    assert report["slot_accounts"][2]["callback_attempted"] == {"packet": 1, "encode": 1}
    assert report["slot_accounts"][2]["callback_completed"] == {"packet": 1, "encode": 0}
    failure = report["slot_accounts"][5]["decision_accounts"][0]
    assert not failure["complete"] and failure["attempted"]["replay_calls"] == 1
    assert failure["completed"]["replay_calls"] == 0
    assert not report["physical_certified"] and not report["scientific_pass"]


@pytest.mark.parametrize("variant", ["wrong_sha", "truncated", "missing_terminal", "bad_chain", "missing_resource",
                                      "changed_slot", "changed_action", "missing_slot_return", "extra_unknown"])
def test_actual_bytes_and_rehashed_tampering_rejected(saved, tmp_path, variant):
    source, digest = saved
    raw = source.read_bytes()
    target = tmp_path / "tampered.jsonl"
    if variant == "wrong_sha":
        target.write_bytes(raw)
        digest = "0" * 64
    elif variant == "truncated":
        target.write_bytes(raw[:-5])
        digest = writer.sha(raw[:-5])
    elif variant == "bad_chain":
        target.write_bytes(raw.replace(b'"index":1,', b'"index":9,', 1))
        digest = writer.sha(target.read_bytes())
    else:
        records = [json.loads(line)["payload"] for line in raw.splitlines()]
        if variant == "missing_terminal":
            records.pop()
        elif variant == "missing_resource":
            index = next(i for i, e in enumerate(records) if e["kind"] == "resource_return")
            records.pop(index)
        elif variant == "changed_slot":
            next(e for e in records if e["kind"] == "slot_return")["row"]["root"] = 1
        elif variant == "changed_action":
            next(e for e in records if e["kind"] == "packet_attempt")["raw"][0][0] += 0.1
        elif variant == "missing_slot_return":
            records.pop(next(i for i, e in enumerate(records) if e["kind"] == "slot_return"))
        else:
            index = next(i for i, e in enumerate(records) if e["kind"] == "slot_final")
            records.insert(index, {"kind": "forged_extra"})
        digest = rewrite(target, records)
    with pytest.raises((ValueError, KeyError, IndexError)):
        reader.consume(target, digest, 128 * 1024 * 1024)


def test_exclusive_and_cap_leave_no_success(tmp_path):
    out = tmp_path / "run"
    store = writer.Store(out, max_bytes=4096, terminal_reserve=500)
    with pytest.raises(ValueError):
        store.event({"kind": "oversized", "value": "x" * 5000})
    with pytest.raises(ValueError, match="unavailable"):
        store.event({"kind": "must_not_resume"})
    with pytest.raises(ValueError):
        writer.Store(out, max_bytes=4096, terminal_reserve=500)
    raw = (out / "journal.jsonl").read_bytes()
    with pytest.raises(ValueError):
        reader.consume(out / "journal.jsonl", writer.sha(raw), 4096)
