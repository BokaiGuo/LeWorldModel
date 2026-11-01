"""Deterministic offline callback simulation; no model, GPU or physics."""
import json

import pytest

from experiments.spectral_reachability import p3_six_preload_accounting_20261030_v1 as a


def setup():
    host = dict(wall=1, cuda_allocated_bytes=100, cuda_reserved_bytes=200)
    plan = dict(caps=dict(wall_ns=2_000_000_000, rss_peak_bytes=1000,
                          cuda_allocated_peak_bytes=100, cuda_reserved_peak_bytes=200,
                          logical_checkpoint_bytes=300),
                host_plan_sha256=a.digest(host), host_wall_ns=1_000_000_000, source_sha256="a"*64,
                journal_cap_bytes=100_000, failure_reserve_bytes=10_000)
    state = dict(tick=0, calls=[], values=dict(rss_peak_bytes=1, cuda_allocated_peak_bytes=0,
                                               cuda_reserved_peak_bytes=0, logical_checkpoint_bytes=0))
    def clock():
        state["tick"] += 10
        return state["tick"]
    def sample(stage, boundary):
        values = state["values"].copy()
        if stage == "cuda_context" and boundary == "before":
            values["cuda_allocated_peak_bytes"] = values["cuda_reserved_peak_bytes"] = None
        return values
    def init(guard):
        state["calls"].append("init")
        guard()
    def load(guard):
        state["calls"].append("load")
        state["values"]["logical_checkpoint_bytes"] = 40
        guard()
        return object()
    def run(model, plan, guard):
        state["calls"].append("host")
        guard()
        return {k: "a"*64 for k in ("begin_sha256", "end_sha256", "report_sha256", "completion_sha256")}
    return plan, host, state, dict(clock_ns=clock, sample=sample, init_context=init, load_model=load, run_host=run)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def execute(path, plan, host, kw):
    return a.run_offline_seam(path, plan, a.digest(plan), host_plan=host, **kw)


def test_full_order_chain_and_unknowns(tmp_path):
    plan, host, state, kw = setup()
    path = tmp_path / "journal"
    result = execute(path, plan, host, kw)
    events = records(path)
    assert state["calls"] == ["init", "load", "host"]
    assert [x["detail"]["stage"] for x in events if x["kind"] == "intent"] == list(a.STAGES)
    for index, event in enumerate(events):
        assert event["index"] == index
        assert event["previous"] == (events[index - 1]["sha256"] if index else None)
        assert event["sha256"] == a.digest({k: v for k, v in event.items() if k != "sha256"})
    assert result["unknown"] == a.UNKNOWN and all(v == "unknown" for v in result["unknown"].values())
    assert not result["scientific_pass"] and not result["accounting_complete"]


@pytest.mark.parametrize("metric", a.METRICS)
def test_budget_overrun_blocks_following_stage_and_records_failure(tmp_path, metric):
    plan, host, state, kw = setup()
    def init(guard):
        state["calls"].append("init")
        state["values"][metric] = plan["caps"][metric] + 1
        guard()
    kw["init_context"] = init
    path = tmp_path / "failure"
    with pytest.raises(ValueError, match="ceiling"):
        execute(path, plan, host, kw)
    assert state["calls"] == ["init"]
    assert records(path)[-1]["kind"] == "failure"


def test_unknown_counter_after_context_fails_closed(tmp_path):
    plan, host, state, kw = setup()
    original = kw["sample"]
    def sample(stage, boundary):
        result = original(stage, boundary)
        if stage == "cuda_context" and boundary == "after":
            result["cuda_reserved_peak_bytes"] = None
        return result
    kw["sample"] = sample
    with pytest.raises(ValueError, match="counter unknown"):
        execute(tmp_path / "unknown", plan, host, kw)
    assert state["calls"] == ["init"]


def test_swallowed_guard_failure_remains_latched(tmp_path):
    plan, host, state, kw = setup()
    def init(guard):
        state["values"]["rss_peak_bytes"] = 1001
        try:
            guard()
        except ValueError:
            state["values"]["rss_peak_bytes"] = 1
    kw["init_context"] = init
    path = tmp_path / "latched"
    with pytest.raises(ValueError, match="previously failed"):
        execute(path, plan, host, kw)
    assert not state["calls"]
    assert records(path)[-1]["kind"] == "failure"


def test_host_wall_must_fit_without_changing_pinned_plan(tmp_path):
    plan, host, state, kw = setup()
    plan["caps"]["wall_ns"] = 1_000_000_001
    with pytest.raises(ValueError, match="insufficient remaining"):
        execute(tmp_path / "wall", plan, host, kw)
    assert state["calls"] == ["init", "load"]


def test_bad_host_pin_or_missing_cap_does_not_create_journal(tmp_path):
    plan, host, state, kw = setup()
    host["wall"] = 2
    path = tmp_path / "wrong"
    with pytest.raises(ValueError, match="host plan"):
        execute(path, plan, host, kw)
    assert not path.exists() and not state["calls"]
    del plan["caps"]["rss_peak_bytes"]
    with pytest.raises(ValueError, match="ceilings"):
        execute(path, plan, host, kw)
    assert not path.exists()


def test_model_load_failure_retains_intent_and_stops_host(tmp_path):
    plan, host, state, kw = setup()
    def fail(guard):
        raise OSError("checkpoint read failed")
    kw["load_model"] = fail
    path = tmp_path / "load_failure"
    with pytest.raises(OSError, match="checkpoint read"):
        execute(path, plan, host, kw)
    events = records(path)
    assert events[-2]["kind"] == "intent" and events[-1]["kind"] == "failure"
    assert events[-1]["detail"]["stage"] == "checkpoint_and_model_load"
    assert state["calls"] == ["init"]


def test_counter_regression_and_nonmonotonic_clock_rejected(tmp_path):
    plan, host, state, kw = setup()
    def load(guard):
        state["values"]["rss_peak_bytes"] = 0
        guard()
    kw["load_model"] = load
    with pytest.raises(ValueError, match="counter regression"):
        execute(tmp_path / "counter", plan, host, kw)
    plan, host, state, kw = setup()
    ticks = iter([10, 9])
    kw["clock_ns"] = lambda: next(ticks)
    with pytest.raises(ValueError, match="monotonic clock"):
        execute(tmp_path / "clock", plan, host, kw)


def test_existing_journal_is_never_overwritten(tmp_path):
    plan, host, state, kw = setup()
    path = tmp_path / "existing"
    path.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="exclusive"):
        execute(path, plan, host, kw)
    assert path.read_bytes() == b"preserve"


def test_invalid_host_output_cannot_complete(tmp_path):
    plan, host, state, kw = setup()
    kw["run_host"] = lambda model, plan, guard: {"report_sha256": "a"*64}
    path = tmp_path / "invalid_host"
    with pytest.raises(ValueError, match="host output pins"):
        execute(path, plan, host, kw)
    assert records(path)[-1]["kind"] == "failure"
    assert all(e["kind"] != "end" for e in records(path))


def test_journal_cap_failure_keeps_reserved_failure_record(tmp_path):
    plan, host, state, kw = setup()
    plan["journal_cap_bytes"] = 2500
    plan["failure_reserve_bytes"] = 1000
    path = tmp_path / "small"
    with pytest.raises(ValueError, match="journal ceiling"):
        execute(path, plan, host, kw)
    assert records(path)[-1]["kind"] == "failure"
    assert path.stat().st_size <= plan["journal_cap_bytes"]
