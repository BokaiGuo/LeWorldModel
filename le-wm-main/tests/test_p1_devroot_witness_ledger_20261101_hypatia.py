import json

import pytest

from experiments.spectral_reachability import p1_devroot_witness_ledger_20261101_hypatia as m


def ledger(path, **kw):
    return m.Ledger(path, min_free_bytes=0, **kw)


def test_attempt_is_durable_before_dispatch_and_failure_retained(tmp_path):
    obj = ledger(tmp_path / "out")

    def fail():
        assert json.loads((obj.directory / "journal.jsonl").read_text())["kind"] == "attempt"
        raise ValueError("physical failure")

    with pytest.raises(ValueError):
        obj.call("reset", fail, slot=0)
    assert obj.counts["reset"] == {"attempted": 1, "completed": 0}
    assert len((obj.directory / "journal.jsonl").read_text().splitlines()) == 1


def test_budget_checks_before_callback(tmp_path):
    obj = ledger(tmp_path / "out")
    obj.counts["warmup_step"]["attempted"] = m.CAPS["warmup_step"]
    called = []
    with pytest.raises(m.GuardError):
        obj.call("warmup_step", lambda: called.append(True))
    assert not called
    assert not (obj.directory / "journal.jsonl").exists()


def test_write_reserve_and_exclusive_output(tmp_path):
    obj = ledger(tmp_path / "out", max_bytes=100, reserve_bytes=20)
    obj.write("one", b"x" * 80)
    with pytest.raises(m.GuardError):
        obj.write("two", b"x")
    with pytest.raises(m.GuardError):
        ledger(tmp_path / "out")


def test_wall_guard_and_chain(tmp_path):
    now = [0]
    obj = ledger(tmp_path / "out", clock=lambda: now[0], wall_seconds=2)
    obj.call("reset", lambda: None)
    events = [json.loads(x) for x in (obj.directory / "journal.jsonl").read_text().splitlines()]
    assert events[1]["previous_sha256"] == events[0]["sha256"]
    now[0] = 2
    with pytest.raises(m.GuardError):
        obj.call("reset", lambda: pytest.fail("must not dispatch"))


def test_sync_failure_poison_stops_further_dispatch(tmp_path, monkeypatch):
    obj = ledger(tmp_path / "out")
    monkeypatch.setattr(m.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        obj.event("attempt", resource="reset")
    assert obj.failed
    with pytest.raises(m.GuardError):
        obj.call("reset", lambda: pytest.fail("must not dispatch"))
