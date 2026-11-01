import os

import pytest

from experiments.spectral_reachability import codex_e1_proposal_durable_events_20261101 as m


def journal(path):
    return m.ProposalJournal(path, run_id="new-run", arm="arm-a", proposal_kind="compiled")


def payloads(path):
    return [row["payload"]["event"] for row in m.read_events(path)]


def test_attempt_visible_before_work_and_return(tmp_path):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        with log.attempt("compile", details={"source": "new"}):
            assert [e["type"] for e in payloads(path)] == ["attempt"]
            log.append({"type": "measurement", "axis": "cpu_seconds", "value": 1.5})
    events = payloads(path)
    assert [e["type"] for e in events] == ["attempt", "measurement", "return"]
    assert events[-1]["attempt_index"] == 0
    assert all(e["payload"]["identity"]["run_id"] == "new-run" for e in m.read_events(path))


def test_failure_preserves_primary_and_paid_attempt(tmp_path):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        with pytest.raises(RuntimeError, match="operation failed"):
            with log.attempt("learn"):
                log.append({"type": "measurement", "axis": "calls", "value": 3})
                raise RuntimeError("operation failed")
    events = payloads(path)
    assert [e["type"] for e in events] == ["attempt", "measurement", "failure"]
    assert events[-1]["attempt_index"] == 0
    assert events[-1]["exception_type"] == "RuntimeError"


def test_existing_file_never_overwritten(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        journal(path)
    assert path.read_bytes() == b"existing"


def test_torn_tail_and_tampering_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        log.append({"type": "test"})
    raw = path.read_bytes()
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError):
        m.read_events(path)
    path.write_bytes(raw.replace(b'"test"', b'"fake"'))
    with pytest.raises(ValueError):
        m.read_events(path)


def test_write_failure_latches_and_does_not_run_work(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        monkeypatch.setattr(m.os, "write", lambda *args: (_ for _ in ()).throw(OSError("disk")))
        with pytest.raises(OSError, match="disk"):
            with log.attempt("compile"):
                pytest.fail("must not execute")
        with pytest.raises(RuntimeError, match="write failed"):
            log.append({"type": "later"})
    assert path.read_bytes() == b""


def test_failure_logging_error_does_not_mask_original(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        with pytest.raises(ValueError, match="primary") as caught:
            with log.attempt("learn"):
                monkeypatch.setattr(m.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk")))
                raise ValueError("primary")
        assert any("could not be persisted" in note for note in caught.value.__notes__)
        assert log.poisoned


def test_append_fsyncs_every_event(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        calls = []
        original = os.fsync
        def sync(fd):
            calls.append(fd)
            original(fd)
        monkeypatch.setattr(m.os, "fsync", sync)
        with log.attempt("compile"):
            pass
        assert calls == [log.fd, log.fd]


def test_invalid_json_fails_before_write(tmp_path):
    path = tmp_path / "events.jsonl"
    with journal(path) as log:
        with pytest.raises(ValueError):
            log.append({"value": float("nan")})
        assert log.index == 0
        log.append({"type": "valid"})
    assert len(m.read_events(path)) == 1
