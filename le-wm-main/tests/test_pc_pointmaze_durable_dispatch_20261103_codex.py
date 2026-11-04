"""Durability ordering and process-crash tests for paired action dispatch."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import pc_pointmaze_durable_dispatch_20261103_codex as durable


class Fake:
    def __init__(self):
        self.calls = 0
        self.clock = 0

    def advance(self, action):
        self.calls += 1
        self.clock += 1
        return np.full((224, 224, 3), self.clock, dtype=np.uint8)


def test_fsync_file_directory_and_each_bridge_event(tmp_path, monkeypatch):
    fsync = os.fsync
    observed = []
    def traced(fd):
        observed.append(os.readlink(f"/proc/self/fd/{fd}"))
        return fsync(fd)
    monkeypatch.setattr(durable.os, "fsync", traced)
    action = np.array([.2, -.3])
    snapshot = tmp_path/"decision.npz"
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        digest = durable.save_snapshot(snapshot, action=action, clock=np.asarray(1))
        a, b = Fake(), Fake()
        durable.dispatch_pair(a, b, action, snapshot, digest, journal, tmp_path/"response", "r", 1, "prefix")
    events = [json.loads(x) for x in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert [(x["event"], x.get("bridge")) for x in events] == [
        ("bridge_attempt", "shadow"), ("bridge_returned", "shadow"),
        ("bridge_attempt", "legacy"), ("bridge_returned", "legacy")]
    assert a.calls == b.calls == 1
    assert observed.count(str(snapshot)) == 1
    assert observed.count(str(tmp_path/"events.jsonl")) == 4
    assert observed.count(str(tmp_path)) >= 4
    assert observed.index(str(snapshot)) < observed.index(str(tmp_path/"events.jsonl"))
    assert all(Path(e["rgb_path"]).exists() for e in events if e["event"] == "bridge_returned")


def test_failed_snapshot_fsync_never_dispatches(tmp_path, monkeypatch):
    a, b = Fake(), Fake()
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        def fail(_):
            raise OSError(errno.EIO, "injected")
        monkeypatch.setattr(durable.os, "fsync", fail)
        with pytest.raises(OSError):
            digest = durable.save_snapshot(tmp_path/"bad.npz", action=np.zeros(2), clock=np.asarray(1))
            durable.dispatch_pair(a, b, np.zeros(2), tmp_path/"bad.npz", digest,
                                  journal, tmp_path/"response", "r", 1, "prefix")
    assert a.calls == b.calls == 0
    assert (tmp_path/"events.jsonl").read_text() == ""


def test_fsync_failed_after_attempt_prevents_dispatch(tmp_path, monkeypatch):
    a, b = Fake(), Fake()
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        path = tmp_path/"decision.npz"
        digest = durable.save_snapshot(path, action=np.zeros(2), clock=np.asarray(1))
        monkeypatch.setattr(durable.os, "fsync", lambda _: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
        with pytest.raises(OSError):
            durable.dispatch_pair(a, b, np.zeros(2), path, digest, journal, tmp_path/"response", "r", 1, "prefix")
    assert a.calls == b.calls == 0


def test_fsync_failed_on_shadow_rgb_prevents_legacy_action(tmp_path, monkeypatch):
    a, b = Fake(), Fake()
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        path = tmp_path/"decision.npz"
        digest = durable.save_snapshot(path, action=np.zeros(2), clock=np.asarray(1))
        actual, count = os.fsync, [0]
        def fail_second(fd):
            count[0] += 1
            if count[0] == 2:
                raise OSError(errno.EIO, "injected response fsync failure")
            return actual(fd)
        monkeypatch.setattr(durable.os, "fsync", fail_second)
        with pytest.raises(OSError):
            durable.dispatch_pair(a, b, np.zeros(2), path, digest, journal, tmp_path/"response", "r", 1, "prefix")
    assert a.calls == 1 and b.calls == 0
    assert [json.loads(x)["event"] for x in (tmp_path/"events.jsonl").read_text().splitlines()] == ["bridge_attempt"]


def test_tampered_snapshot_rejected_before_dispatch(tmp_path):
    a, b = Fake(), Fake()
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        path = tmp_path/"decision.npz"
        digest = durable.save_snapshot(path, action=np.zeros(2), clock=np.asarray(1))
        with pytest.raises(ValueError, match="action mismatch"):
            durable.dispatch_pair(a, b, np.ones(2), path, digest, journal, tmp_path/"response", "r", 1, "prefix")
    assert a.calls == b.calls == 0


def test_reset_has_individual_durable_attempt_and_return(tmp_path):
    class ResetEnv:
        step_count = 0
        initial_rgb = np.zeros((224, 224, 3), np.uint8)
        def close(self):
            pass
    with durable.DurableJournal(tmp_path/"events.jsonl") as journal:
        path = tmp_path/"reset.npz"
        digest = durable.save_snapshot(path, clock=np.asarray(0), source_index=np.asarray(4))
        durable.dispatch_reset("shadow", ResetEnv, path, digest, journal, tmp_path/"reset_response", "r")
    rows = [json.loads(x) for x in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["bridge_attempt", "bridge_returned"]
    assert rows[0]["phase"] == rows[1]["phase"] == "reset"
    assert durable.sha(rows[1]["rgb_path"]) == rows[1]["artifact_sha256"]


@pytest.mark.parametrize("stage,expected", [
    ("after_shadow", [("bridge_attempt", "shadow"), ("bridge_returned", "shadow")]),
    ("during_legacy", [("bridge_attempt", "shadow"), ("bridge_returned", "shadow"),
                       ("bridge_attempt", "legacy")]),
])
def test_abrupt_process_exit_preserves_exact_bridge_boundary(tmp_path, stage, expected):
    script = Path(__file__).parent/"fixtures/pc_pointmaze_durable_crash_20261103_codex.py"
    proc = subprocess.run([sys.executable, str(script), str(tmp_path), stage], env=os.environ.copy())
    assert proc.returncode == 73
    rows = [json.loads(x) for x in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert [(x["event"], x.get("bridge")) for x in rows] == expected
    assert (tmp_path/"decision.npz").is_file()
    assert (tmp_path/"response_shadow.npz").is_file()
    assert not (tmp_path/"response_legacy.npz").exists()
