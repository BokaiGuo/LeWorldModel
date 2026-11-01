import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_paired_durable_journal_20261030.py"
spec = importlib.util.spec_from_file_location("durable_pair", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = m.load_base()
A, B = base.ARMS
H, J = "a" * 64, "b" * 64
LABEL = {"state7": [0.] * 7, "reward": -1.}


def config():
    identity = {k: H for k in base.IDENTITY}
    return {"identities": {a: copy.deepcopy(identity) for a in base.ARMS},
            "initial_frames": {a: [H, H, H] for a in base.ARMS},
            "initial_actions": {a: [[[0., 0.]], [[0., 0.]]] for a in base.ARMS},
            "horizon": 2, "packet_steps": 1,
            "caps": {a: {k: 20 for k in base.AXES} for a in base.ARMS}}


def select(j, arm):
    s = j.protocol.state(arm)
    req = j.call("observe", arm, s["step"], s["frames"], s["actions"])
    return j.call("select", arm, req, [[0.1, 0.2]], H)


def complete(j, arm, pixel=J, terminated=False):
    token = select(j, arm)
    event = j.call("begin_step", arm, token, True)
    j.call("finish_step", arm, event, pixel, LABEL, terminated=terminated)


def build(path):
    j = m.Journal(path, config())
    j.call("record_resource_attempt", A, "p0", {"predict_calls": 2})
    j.call("record_resource_return", A, "p0")
    complete(j, A)
    complete(j, A)
    complete(j, B, terminated=True)
    return j.finish()


def rechain(records):
    tail = "0" * 64
    output = []
    for i, record in enumerate(records):
        body = {"seq": i, "prev": tail, "kind": record["kind"], "payload": record["payload"]}
        tail = m.hashbytes(m.canonical(body))
        output.append(m.canonical({**body, "sha256": tail}) + b"\n")
    return b"".join(output)


def test_complete_and_early_stop_rebuild(tmp_path):
    path = tmp_path / "events.jsonl"
    sha = build(path)
    out = m.consume(path, sha)
    rows = out["snapshot"]["rows"]
    assert rows[0]["status"] == "complete" and rows[1]["status"] == "stopped"
    assert rows[1]["checkpoints"]["2"] is None
    assert rows[0]["resources_attempted"]["predict_calls"] == 2
    assert out["snapshot"]["equal_total_compute"] is False
    assert out["new_physical_steps"] == 0 and not out["snapshot"]["actual_execution_verified"]
    kinds = [json.loads(x)["kind"] for x in path.read_bytes().splitlines()]
    assert kinds.count("seal") == 6


def test_external_anchor_and_exclusive(tmp_path):
    path = tmp_path / "events.jsonl"
    sha = build(path)
    with pytest.raises(FileExistsError):
        m.Journal(path, config())
    with pytest.raises(ValueError, match="hash mismatch"):
        m.consume(path, "0" * 64)
    assert m.consume(path, sha)["journal_sha256"] == sha


def test_truncated_and_incomplete_refused(tmp_path):
    path = tmp_path / "partial"
    j = m.Journal(path, config())
    j.call("stop", A, "not_run")
    j.close_partial()
    raw = path.read_bytes()
    with pytest.raises(ValueError, match="footer"):
        m.consume(path, m.hashbytes(raw))
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="truncated"):
        m.consume(path, m.hashbytes(path.read_bytes()))


def test_failed_call_poisoned_no_retry(tmp_path):
    path = tmp_path / "failed"
    j = m.Journal(path, config())
    with pytest.raises(ValueError):
        j.call("begin_step", A, H, True)
    with pytest.raises(ValueError, match="poisoned"):
        j.call("stop", A, "retry")
    j.close_partial()
    with pytest.raises(ValueError, match="footer"):
        m.consume(path, m.hashbytes(path.read_bytes()))


def test_pending_physical_and_resource_survive(tmp_path):
    path = tmp_path / "pending"
    j = m.Journal(path, config())
    j.call("record_resource_attempt", A, "predict", {"predict_calls": 1})
    j.call("stop", A, "unknown")
    token = select(j, B)
    j.call("begin_step", B, token, True)
    j.call("stop", B, "unknown_step_return")
    out = m.consume(path, j.finish())
    a, b = out["snapshot"]["rows"]
    assert a["pending_resources"] == {"predict": {"predict_calls": 1}}
    assert b["physical_step_pending"] and b["resources_attempted"]["physical_steps"] == 1
    assert b["resources_completed_lower_bound"]["physical_steps"] == 0 and b["checkpoints"]["1"] is None


@pytest.mark.parametrize("tamper", ["seal", "result", "snapshot", "source", "reorder", "extra"])
def test_resealed_tampering_rejected(tmp_path, tamper):
    path = tmp_path / "tamper"
    build(path)
    events = [json.loads(x) for x in path.read_bytes().splitlines()]
    if tamper == "seal":
        next(e for e in events if e["kind"] == "seal")["payload"]["content_sha256"] = "0" * 64
    elif tamper == "result":
        next(e for e in events if e["kind"] == "commit")["payload"]["result"] = "false result"
    elif tamper == "snapshot":
        events[-1]["payload"]["snapshot"]["fixed_time_denominator"] = 1
    elif tamper == "source":
        events[0]["payload"]["base_sha256"] = "0" * 64
    elif tamper == "reorder":
        i = next(i for i, e in enumerate(events) if e["kind"] == "seal")
        events[i], events[i+1] = events[i+1], events[i]
    else:
        events.insert(-1, {"kind": "unknown", "payload": {}})
    raw = rechain(events)
    path.write_bytes(raw)
    with pytest.raises((ValueError, TypeError)):
        m.consume(path, m.hashbytes(raw))


def test_fsync_failure_poisoned(tmp_path, monkeypatch):
    path = tmp_path / "io"
    j = m.Journal(path, config())
    def fail(_):
        raise OSError("fixture fsync failure")
    monkeypatch.setattr(m.os, "fsync", fail)
    with pytest.raises(OSError):
        j.call("stop", A, "x")
    with pytest.raises(ValueError, match="poisoned"):
        j.finish()
    j.close_partial()


def test_operation_and_footer_guards(tmp_path):
    path = tmp_path / "guards"
    j = m.Journal(path, config())
    with pytest.raises(ValueError, match="forbidden"):
        j.call("snapshot")
    with pytest.raises(ValueError, match="terminal"):
        j.finish()
    j.call("stop", A, "unstarted")
    j.call("stop", B, "unstarted")
    assert m.consume(path, j.finish())["snapshot"]["fixed_arm_denominator"] == 2


def test_consumer_cli_exclusive_and_bad_hash_no_output(tmp_path, monkeypatch):
    path, out = tmp_path / "events", tmp_path / "receipt.json"
    sha = build(path)
    monkeypatch.setattr(sys, "argv", ["consume", "--journal", str(path), "--sha256", "0"*64, "--output", str(out)])
    with pytest.raises(ValueError, match="hash mismatch"):
        m.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", ["consume", "--journal", str(path), "--sha256", sha, "--output", str(out)])
    m.main()
    assert json.loads(out.read_text())["journal_sha256"] == sha
    with pytest.raises(FileExistsError):
        m.main()
