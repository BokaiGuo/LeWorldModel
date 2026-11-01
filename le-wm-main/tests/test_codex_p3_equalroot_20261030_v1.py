import copy
import json
import pytest

from experiments.spectral_reachability import codex_p3_equalroot_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p3_equalroot_consumer_20261030_v1 as consumer
from experiments.spectral_reachability import codex_p3_equalroot_schedule_20261030_v1 as schedule
from test_codex_p3_quota_native_bridge_20261030_v1 import Backend, inputs


def test_bank_and_equal_queue_are_fixed():
    assert schedule.banks(1.0) == schedule.banks(1.0)
    for weights in ([2., 1.], [1., 2.], [1., 1.]):
        for policy in schedule.POLICIES:
            q = schedule.queue(policy, weights)
            for root in (0, 1):
                values = [j["index"] for j in q["jobs"] if j["root"] == root]
                assert len(values) == len(set(values)) == 10
    assert schedule.queue("signal", [2., 1.])["jobs"] != schedule.queue("reverse", [2., 1.])["jobs"]
    with pytest.raises(ValueError):
        schedule.queue("signal", [float("nan"), 1.])


def test_backend_queue_consumer_and_tamper(monkeypatch):
    tasks, banks = inputs()
    events = []
    meter = bridge.Meter(events.append, lambda: None, lambda: None)
    bridge._execute(tasks, banks, [2., 1.], [Backend(), Backend()], meter,
                    check_path=lambda *args: None, cost=lambda task, path: -path[1][0][0])
    events = json.loads(json.dumps(events))
    monkeypatch.setattr(consumer.native, "check_path", lambda *args: None)
    monkeypatch.setattr(consumer.native, "cost", lambda task, path: -path[1][0][0])
    result = consumer.verify_events(events, tasks, banks, [2., 1.])
    assert result["candidates"] == 60 and result["operation_entries"] == 480
    for policy in schedule.POLICIES:
        for root in (0, 1):
            for axis in bridge.AXES:
                assert sum(e["resource"] == axis and e["scope"][:2] == [policy, root]
                           for e in meter.events) == 20
    for kind, key, value in [("fixed_roster", "candidate_orders", {}),
                             ("metric_return", "value", 999),
                             ("candidate_complete", "row", {}),
                             ("arm_complete", "winners", [])]:
        bad = copy.deepcopy(events)
        next(x for x in bad if x["kind"] == kind)[key] = value
        with pytest.raises((ValueError, KeyError)):
            consumer.verify_events(bad, tasks, banks, [2., 1.])


def test_failed_candidate_no_retry():
    tasks, banks = inputs()
    events = []
    meter = bridge.Meter(events.append, lambda: None, lambda: None)
    with pytest.raises(RuntimeError):
        bridge._execute(tasks, banks, [2., 1.], [Backend(fail=0), Backend()], meter,
                        check_path=lambda *args: None, cost=lambda *args: 0.)
    assert events[-1]["kind"] == "panel_failure"
    assert events[-1]["retry_authorized"] is False
