import pytest

from experiments.spectral_reachability import p3_distinct_host_adapter_20261102_codex_exclusive as a


def allocation(point=20, arm="signal", quota=None):
    return {"kind": "allocation_frozen", "scope": [point, arm, "search", 0],
            "quota": [15, 5] if quota is None else quota, "weights": [2.0, 1.0]}


def test_real_sealed_existing_journal_is_distinct_and_checked():
    _report, events, _ = a.existing.verify_primary_run(a.existing.PRIMARY_RUN, a.PREVIOUS_REPORT)
    result = a.check_events(events)
    assert result["status"] == "quota_monitor_complete"
    assert result["P3_admission"] is False


def test_rejected_allocation_is_durably_saved_before_any_candidate(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = a.host.journal.Journal(path, 100_000, 10_000)
    monitor = a.QuotaMonitor(journal.event)
    monitor(allocation(20, "uniform", [12, 12]))
    # This synthetic sequence stops before completing uniform, which itself must
    # prevent a later allocation from being accepted.
    with pytest.raises(ValueError, match="allocation lifecycle"):
        monitor(allocation())
    journal.append({"kind": "host_failure", "error": "allocation lifecycle"}, terminal=True)
    tail = journal.tail
    journal.close()
    events = a.host.journal.read(path, tail, 100_000)
    assert [e["kind"] for e in events] == ["allocation_frozen", "allocation_frozen", "host_failure"]
    assert not any(e["kind"] == "candidate_attempt" for e in events)


def test_actual_meter_failure_preserves_unreturned_attempt(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = a.host.journal.Journal(path, 100_000, 10_000)
    meter = a.host.bridge.Meter(a.QuotaMonitor(journal.event), lambda: None, lambda: None)
    meter.scope = [20, "signal", "search", 0]
    def fail():
        raise RuntimeError("synthetic dispatch failure")
    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        meter.call("predict", fail)
    journal.append({"kind": "host_failure", "error": "synthetic dispatch failure"}, terminal=True)
    tail = journal.tail
    journal.close()
    events = a.host.journal.read(path, tail, 100_000)
    attempts = [e for e in events if e["kind"] == "resource_attempt"]
    returns = [e for e in events if e["kind"] == "resource_return"]
    assert len(attempts) == 1 and attempts[0]["entry"]["resource"] == "predict"
    assert not returns and events[-1]["kind"] == "host_failure"


def test_duplicate_run_requires_both_authorities():
    with pytest.raises(ValueError, match="authority"):
        a.launch("missing", "missing", "missing", authority=None, repeat_known_panel=False)
    with pytest.raises(ValueError, match="authority"):
        a.launch("missing", "missing", "missing", authority=a.AUTHORITY, repeat_known_panel=False)
