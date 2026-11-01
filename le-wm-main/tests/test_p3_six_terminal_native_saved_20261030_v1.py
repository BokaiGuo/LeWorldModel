"""Synthetic CPU journal, never real native execution."""
from copy import deepcopy
import os

import pytest

import test_p3_six_durable_constructor_20261030_v1 as fixture
from experiments.spectral_reachability import p3_six_terminal_20261030_v1 as terminal
from experiments.spectral_reachability import p3_six_native_saved_20261030_v1 as saved


def start(store):
    rows = terminal.roster([])
    store.event("physical", "record", "six_denominator", store.index,
                dict(rows=rows, fixed_denominator=6))
    store.event("physical", "record", "six_slot_attempt", store.index,
                dict(slot=0, root=0, policy="fixed_order"))


def meter_call(store, resource, region):
    detail = dict(episode_id="matched_slot_00", region=region, units=1)
    token = store.index
    store.event("physical", "intent", "native_meter/" + resource, token, detail)
    store.event("physical", "return", "native_meter/" + resource, token, detail)


def finish_slot(store, status="native_stopped"):
    row = terminal.roster([])[0]
    row.update(status=status, error=None)
    store.event("physical", "record", "six_slot_return", store.index, row)


def test_partial_publish_and_saved_native_meter(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    meter_call(store, "constructor", "setup")
    meter_call(store, "public_step", "prefix")
    raw = "0000000000000000"
    detail = dict(episode_id="matched_slot_00", step=1, raw_bits=raw)
    store.event("physical", "record", "native_authorized_dispatch", store.index, detail)
    store.event("physical", "record", "native_control_return", store.index,
                dict(**detail, terminated=True, truncated=False))
    meter_call(store, "close", "cleanup")
    finish_slot(store)
    hashes = terminal.publish(store)
    result = saved.consume(store.output, hashes["report_sha256"], hashes["completion_sha256"])
    assert result["rows"][0]["status"] == "native_stopped"
    assert all(r["status"] == "not_started" for r in result["rows"][1:])
    assert result["slots"][0]["native_stopped"] and result["slots"][0]["returned_actions"] == 1
    assert not result["full_saved_strict"] and not result["scientific_pass"]
    with pytest.raises(ValueError):
        terminal.publish(store)


def test_report_and_artifact_tampering_rejected(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    store.write("unreferenced.npy", b"fixture", "numerical_arrays")
    hashes = terminal.publish(store)
    result = saved.consume(store.output, hashes["report_sha256"], hashes["completion_sha256"])
    assert result["artifacts"]["orphans"] == ["unreferenced.npy"]
    (store.output / "unreferenced.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory"):
        saved.consume(store.output, hashes["report_sha256"], hashes["completion_sha256"])
    with pytest.raises(ValueError, match="terminal hash"):
        saved.consume(store.output, "0" * 64, hashes["completion_sha256"])


def test_pending_and_torn_tail_preserved(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    detail = dict(episode_id="matched_slot_00", region="setup", units=1)
    store.event("physical", "intent", "native_meter/constructor", store.index, detail)
    with (store.output / "events.jsonl").open("ab") as f:
        f.write(b'{"partial":')
    store.poisoned = True
    hashes = terminal.publish(store)
    result = saved.consume(store.output, hashes["report_sha256"], hashes["completion_sha256"])
    assert result["trailing_bytes"] == len(b'{"partial":')
    assert result["rows"][0]["status"] == "attempted_unsettled"
    assert result["slots"][0]["pending_meter"] and not result["journal_transport_complete"]


def test_native_meter_action_and_stop_tampering_rejected(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    meter_call(store, "public_step", "prefix")
    detail = dict(episode_id="matched_slot_00", step=1, raw_bits="0000000000000000")
    store.event("physical", "record", "native_authorized_dispatch", store.index, detail)
    store.event("physical", "record", "native_control_return", store.index,
                dict(**detail, terminated=True, truncated=False))
    events = fixture.audit(store)["events"][2:]
    tampered = deepcopy(events)
    tampered[-1]["detail"]["raw_bits"] = "0000803f00000000"
    with pytest.raises(ValueError, match="raw binding"):
        saved._native_slot(tampered, 0)
    tampered = deepcopy(events)
    tampered.append(dict(tampered[-2], index=99, detail=dict(detail, step=2)))
    with pytest.raises(ValueError, match="ordered native dispatch"):
        saved._native_slot(tampered, 0)
    tampered = deepcopy(events)
    tampered[1]["detail"]["episode_id"] = "matched_slot_01"
    with pytest.raises(ValueError, match="meter ABI"):
        saved._native_slot(tampered, 0)


def test_roster_rejects_reordering_and_missing_denominator(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    events = fixture.audit(store)["events"]
    changed = deepcopy(events)
    changed[1]["detail"]["slot"] = 3
    with pytest.raises(ValueError, match="ordered attempt"):
        terminal.roster(changed)
    with pytest.raises(ValueError, match="fixed denominator"):
        terminal.roster(events[1:])


def test_no_terminal_when_reserve_insufficient(tmp_path):
    store = fixture.new_store(tmp_path)
    start(store)
    store.reserve = 1
    with pytest.raises(ValueError, match="reserve"):
        terminal.publish(store)
    assert not (store.output / terminal.COMPLETION).exists()
    assert store.sealed


def test_completion_fsync_failure_never_returns_success(tmp_path, monkeypatch):
    store = fixture.new_store(tmp_path)
    start(store)
    original = os.fsync
    calls = []
    def fail_third(fd):
        calls.append(fd)
        if len(calls) == 3:
            raise OSError("terminal fsync")
        return original(fd)
    monkeypatch.setattr(os, "fsync", fail_third)
    with pytest.raises(OSError, match="terminal fsync"):
        terminal.publish(store)
    assert (store.output / terminal.REPORT).is_file()
    assert store.poisoned and store.sealed
    with pytest.raises(ValueError):
        terminal.publish(store)


def test_complete_action_and_encoding_binding_rejects_tampering(tmp_path, monkeypatch):
    counts = {k: dict(attempted=0, returned=0, failed=0) for k in saved.meter.CAPS}
    for key, n in (("constructor", 1), ("close", 1), ("reset", 2), ("goal_bind", 1), ("public_step", 30)):
        counts[key].update(attempted=n, returned=n)
    monkeypatch.setattr(saved, "_native_slot", lambda events, slot: dict(
        native_counts=counts, pending_meter=[], native_stopped=False, closed=True))
    monkeypatch.setattr(saved.encoder, "inspect", lambda *args: None)
    events = []
    def add(origin, kind, resource, detail):
        events.append(dict(index=len(events), origin=origin, kind=kind, resource=resource, detail=detail))
    def observe(phase, step):
        frame = dict(file=f"{phase}_{step}.png", sha256="a" * 64, pixel_sha256="b" * 64)
        receipt = dict(phase=phase, step=step, path=str(tmp_path / frame["file"]),
                       file_sha256=frame["sha256"], pixel_sha256=frame["pixel_sha256"])
        add("physical", "record", "native_fresh_observation", dict(receipt=receipt))
        return frame
    for i in range(11):
        observe("warmup", i)
    observe("plan", 0)
    for step in range(1, 21):
        add("physical", "record", "native_authorized_dispatch", dict(raw_bits="0000000000000000"))
        frame = observe("plan", step)
        add("physical", "record", "native_control_return", dict(raw_bits="0000000000000000"))
        if step % 5 == 0:
            for kind, resource in (("intent", "encoder"), ("return", "encoder"),
                                   ("intent", "projector"), ("return", "projector")):
                add("encoding", kind, resource, dict(boundary=step))
            add("encoding", "record", "encoding_complete", dict(boundary=step, frame=frame, arrays={}))
    result = dict(packets=[dict(raw=[[0.0, 0.0]] * 5) for _ in range(4)])
    assert saved._complete_slot(events, 0, result, tmp_path)["native_action_packet_binding"]
    bad = deepcopy(result)
    bad["packets"][0]["raw"][0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="packet/native"):
        saved._complete_slot(events, 0, bad, tmp_path)
    bad_events = deepcopy(events)
    encoding = next(e for e in bad_events if e["resource"] == "encoding_complete")
    encoding["detail"]["frame"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="frame/encoding"):
        saved._complete_slot(bad_events, 0, result, tmp_path)
