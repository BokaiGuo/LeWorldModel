"""Saved-only CPU fixtures; no CUDA, model inference, or physical execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_decision_dual_branch_20261030_codex_v1 as dual
from experiments.spectral_reachability import p3_dual_branch_saved_consumer_20261030_codex_v1 as c

helper = runpy.run_path(str(dual.core.ROOT / "tests/test_agent_p3_r_native_orchestrator_20261029_v1.py"))


def decision(branch):
    return dict(continue_expanding=True, program_type=branch, extension_type=branch,
                candidates=4 if branch == "progress" else 8, horizon=3)


@pytest.fixture(scope="module", params=["progress", "reconfiguration"])
def stopped(request):
    task = helper["r_task"]()
    store = helper["Memory"]()
    original = dual.core.choose
    dual.core.choose = lambda *args: decision(request.param)
    class Feedback:
        episode_id = "fixture-episode"
        def packet(self, start, raw):
            return dict(raw=raw, rgb_bytes=b"", end=start + 5, stopped=True, episode_id=self.episode_id)
    try:
        result = dual.execute_slot(task, "fixed_h1", np.zeros((5, 2), np.float32), Feedback(),
            lambda t, s: helper["NonlinearRAM"](t, s), store,
            lambda stage, cid, call: store.capsule(stage, cid, call),
            kind="RAM_FIXTURE", permit_ram=True)
    finally:
        dual.core.choose = original
    # Fixture forces branch; restore its scheduler only while auditing it.
    return request.param, result, store.events


def audit_fixture(monkeypatch, stopped):
    branch, result, events = stopped
    monkeypatch.setattr(dual.core, "choose", lambda *args: decision(branch))
    row = dict(policy="fixed_h1", status=result["status"], result=result)
    account = c.audit_slot(row, [{"payload": e} for e in events])
    return row, account


def test_both_real_branch_records_audited_and_stop_preserved(monkeypatch, stopped):
    row, account = audit_fixture(monkeypatch, stopped)
    assert row["status"] == "native_stopped"
    assert account["decisions"] == account["packets"] == 1
    assert account["callback_completed"] == {"packet": 1, "encode": 0}


@pytest.mark.parametrize("target", ["decision", "branch", "budget", "action", "resource"])
def test_semantic_tamper_rejected(monkeypatch, stopped, target):
    branch, result, source = stopped
    events = deepcopy(source)
    result = deepcopy(result)
    monkeypatch.setattr(dual.core, "choose", lambda *args: decision(branch))
    if target == "action":
        next(e for e in events if e["kind"] == "packet_attempt")["raw"][0][0] += 0.1
    elif target == "resource":
        index = next(i for i, e in enumerate(events) if e["kind"] == "io" and
                     e["payload"].get("kind") == "resource_return")
        events.pop(index)
    else:
        final = next(e["payload"]["result"] for e in events if e["kind"] == "io" and
                     e["payload"].get("kind") == "dual_decision_final")
        block = result["decisions"][0]
        if target == "decision":
            final["decision"]["horizon"] = block["decision"]["horizon"] = 1
        elif target == "branch":
            final["branch"] = block["branch"] = "wrong"
        else:
            final["resource_receipt"]["caps"]["predict_API"] += 1
            block["resource_receipt"]["caps"]["predict_API"] += 1
        events[-1]["result"] = result
    with pytest.raises((ValueError, KeyError, IndexError)):
        c.audit_slot(dict(policy="fixed_h1", status=result["status"], result=result),
                     [{"payload": e} for e in events])


def native_events(raw):
    episode = "matched_slot_00"
    events = []
    for i, command in enumerate(raw):
        bits = np.asarray(command, dtype="<f4").tobytes().hex()
        for resource in ("native_authorized_dispatch", "native_control_return"):
            detail = dict(episode_id=episode, step=i + 1, raw_bits=bits)
            if resource == "native_authorized_dispatch":
                detail["packet_boundary"] = 0
            else:
                detail.update(terminated=i == 4, truncated=False)
            events.append(dict(index=len(events), origin="physical", resource=resource,
                               kind="record", token=len(events), detail=detail))
    return events


def test_native_action_binding_and_tamper(monkeypatch, stopped, tmp_path):
    row, account = audit_fixture(monkeypatch, stopped)
    records = native_events(row["result"]["packets"][0]["raw"])
    result = c.audit_native_binding(0, records, account, tmp_path)
    assert result["returns_bound"] == result["dispatches_bound"] == 5
    records[0]["detail"]["raw_bits"] = "000000000000803f"
    with pytest.raises(ValueError):
        c.audit_native_binding(0, records, account, tmp_path)


def test_refuse_active_service_before_reading_files():
    calls = []
    def active(unit):
        calls.append(unit)
        raise ValueError("active")
    with pytest.raises(ValueError, match="active"):
        c.consume(Path("/nonexistent"), "0" * 64, "fixture.service", 100, 100, check_service=active)
    assert calls == ["fixture.service"]


@pytest.mark.parametrize("state,pid,group,ok", [
    ("active", "7", "/fixture", False),
    ("inactive", "0", "/fixture", True),
    ("inactive", "0", "/fixture_busy", False),
    ("failed", "0", "/fixture", True),
])
def test_systemd_liveness_gate(tmp_path, state, pid, group, ok):
    output = f"LoadState=loaded\nActiveState={state}\nSubState=dead\nMainPID={pid}\nControlGroup={group}\nResult=success\n"
    runner = lambda *args, **kwargs: SimpleNamespace(stdout=output)
    if group:
        group_path = tmp_path / group.lstrip("/")
        group_path.mkdir()
        (group_path / "cgroup.procs").write_text("" if "busy" not in group else "123\n")
    if ok:
        assert c.service_stopped("fixture.service", runner, tmp_path)["ActiveState"] == state
    else:
        with pytest.raises(ValueError):
            c.service_stopped("fixture.service", runner, tmp_path)


def test_journal_hash_chain_and_incomplete_terminal(tmp_path):
    p = tmp_path / "journal.jsonl"
    body = dict(index=0, previous=None, payload=dict(kind="journal_begin"))
    record = dict(body, sha256=dual.core.digest(body))
    raw = dual.core.canonical(record) + b"\n"
    p.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    assert len(c.read_journal(p, sha, 10000)) == 1
    p.write_bytes(raw[:-1])
    with pytest.raises(ValueError):
        c.read_journal(p, hashlib.sha256(raw[:-1]).hexdigest(), 10000)
    p.write_bytes(raw.replace(b'"index":0', b'"index":9'))
    with pytest.raises(ValueError, match="journal chain"):
        c.read_journal(p, hashlib.sha256(p.read_bytes()).hexdigest(), 10000)


def test_failed_original_decision_preserved(monkeypatch):
    task = helper["r_task"]()
    store = helper["Memory"]()
    rejected = dict(decision("progress"), program_type="composition")
    monkeypatch.setattr(dual.core, "choose", lambda *args: rejected)
    class Feedback:
        episode_id = "fixture"
        def packet(self, *args):
            raise AssertionError("failed decision must not dispatch")
    result = dual.execute_slot(task, "fixed_h1", np.zeros((5, 2), np.float32), Feedback(),
        lambda t, s: helper["NonlinearRAM"](t, s), store,
        lambda stage, cid, call: store.capsule(stage, cid, call), kind="RAM_FIXTURE", permit_ram=True)
    account = c.audit_slot(dict(policy="fixed_h1", status=result["status"], result=result),
                           [{"payload": e} for e in store.events])
    assert account["status"] == "failed" and account["packets"] == 0
    assert result["decisions"][0]["decision"] == rejected and result["decisions"][0]["branch"] is None


def encode_decision_records(payloads):
    output, prev, hashes = [], None, []
    for i, payload in enumerate(payloads):
        body = dict(index=i, previous=prev, payload=payload)
        entry = dict(body, sha256=dual.core.digest(body))
        output.append(dual.core.canonical(entry) + b"\n")
        prev = entry["sha256"]
        hashes.append(prev)
    return b"".join(output), hashes


def encode_native_records(events):
    output, prev = [], None
    for i, event in enumerate(events):
        body = dict(index=i, previous=prev, origin=event["origin"], kind=event["kind"],
                    resource=event["resource"], token=event["token"], detail=event["detail"])
        entry = dict(body, sha256=c.transport.base.digest(body))
        output.append(c.transport.base.raw(entry) + b"\n")
        prev = entry["sha256"]
    return b"".join(output), prev


def test_end_to_end_six_rows_native_anchor_and_rehashed_attack(monkeypatch, stopped, tmp_path):
    branch, result, detail = stopped
    monkeypatch.setattr(dual.core, "choose", lambda *args: decision(branch))
    out = tmp_path / "run"
    (out / "decision").mkdir(parents=True)
    (out / "native").mkdir()
    initial = [dict(slot=s, root=r, policy=p, status="not_started", result=None,
                    error=None, cleanup="not_required") for s, r, p in dual.ROSTER]
    first = dict(slot=0, root=0, policy="fixed_h1", status=result["status"],
                 result=result, error=None, cleanup="returned")
    rows = [first] + [dict(slot=s, root=r, policy=p, status="setup_failed", result=None,
                           error="RuntimeError: setup", cleanup="not_required")
                      for s, r, p in dual.ROSTER[1:]]
    summary = [{k: row[k] for k in ("slot", "root", "policy", "status", "cleanup", "error")} for row in rows]
    payloads = [dict(kind="journal_begin", pins={"fixture": True}),
                dict(kind="dual_fixed_denominator", rows=initial)]
    positions = []
    for row in rows:
        positions.append(len(payloads))
        payloads.append(dict(kind="dual_slot_attempt", slot=row["slot"], root=row["root"], policy=row["policy"]))
        if row["slot"] == 0:
            payloads.extend(deepcopy(detail))
        payloads.append(dict(kind="dual_slot_return", row=row))
    stop = positions[1] - 1
    payloads.append(dict(kind="dual_journal_end", rows=summary, pins={"fixture": True}))
    journal_data, hashes = encode_decision_records(payloads)
    (out / "decision/journal.jsonl").write_bytes(journal_data)
    anchor = lambda resource, data: dict(origin="physical", kind="record", resource=resource,
                                         token=0, detail=data)
    begin = anchor(c.ADAPTER + "_slot_begin", dict(slot=0, root=0, policy="fixed_h1",
                   decision_index=positions[0] + 1, decision_tail=hashes[positions[0]]))
    finish = anchor(c.ADAPTER + "_slot_end", dict(slot=0, root=0, policy="fixed_h1",
                    decision_index=stop, decision_tail=hashes[stop - 1], cleanup="returned"))
    events = [begin] + native_events(result["packets"][0]["raw"]) + [finish]
    data, tail = encode_native_records(events)
    native_path = out / "native/events.jsonl"
    native_path.write_bytes(data)
    native_sha = hashlib.sha256(data).hexdigest()
    end = dict(schema=c.DRIVER + ".end", full_saved_strict=False, physical_certified=False,
               scientific_pass=False, decision_sha256=hashlib.sha256(journal_data).hexdigest(),
               rows=summary, native_files=[dict(name="events.jsonl", bytes=len(data), sha256=native_sha)],
               native_index=len(events), native_tail=tail, native_pending=0)
    raw = dual.core.canonical(end) + b"\n"
    (out / "end.json").write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    checker = lambda unit: {"status": "offline fixture"}
    report = c.consume(out, checksum, "fixture.service", 256 * 1024**2, 1024**2, check_service=checker)
    assert [r["status"] for r in report["rows"]] == ["native_stopped"] + ["setup_failed"] * 5
    assert report["native_accounts"][0]["returns_bound"] == 5 and not report["scientific_pass"]
    # Fresh outer file hashes do not repair an anchor changed inside the native chain.
    events[-1]["detail"]["decision_tail"] = "0" * 64
    changed, tail = encode_native_records(events)
    native_path.write_bytes(changed)
    end["native_files"][0].update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    end["native_tail"] = tail
    raw = dual.core.canonical(end) + b"\n"
    (out / "end.json").write_bytes(raw)
    with pytest.raises(ValueError, match="end anchor"):
        c.consume(out, hashlib.sha256(raw).hexdigest(), "fixture.service", 256 * 1024**2,
                  1024**2, check_service=checker)
