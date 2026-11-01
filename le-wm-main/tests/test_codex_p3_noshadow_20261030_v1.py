import copy
import json
import pytest
from experiments.spectral_reachability import codex_p3_noshadow_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import codex_p3_noshadow_consumer_20261030_v1 as consumer
from experiments.spectral_reachability import codex_p3_noshadow_host_20261030_v1 as host
from test_codex_p3_quota_native_bridge_20261030_v1 import Backend, inputs


def fixture(monkeypatch):
    tasks, banks = inputs()
    events = []
    meter = bridge.Meter(events.append, lambda: None, lambda: None)
    bridge._execute(tasks, banks, [2., 1.], [Backend(), Backend()], meter,
                    check_path=lambda task, raw, path: None,
                    cost=lambda task, path: -path[1][0][0])
    monkeypatch.setattr(consumer.native, "check_path", lambda task, raw, path: None)
    monkeypatch.setattr(consumer.native, "cost", lambda task, path: -path[1][0][0])
    return json.loads(json.dumps(events)), tasks, banks


def test_real_queue_no_padding_and_saved_tampering(monkeypatch):
    events, tasks, banks = fixture(monkeypatch)
    result = consumer.verify_events(events, tasks, banks, [2., 1.])
    assert result["candidates"] == 60 and result["operation_entries"] == 480
    assert [[w["index"] for w in arm] for arm in result["winners"]] == [[9, 9], [14, 4], [4, 14]]
    for kind, key, value in [("metric_return", "value", 999),
                             ("candidate_complete", "row", {}),
                             ("arm_complete", "winners", [])]:
        bad = copy.deepcopy(events)
        next(x for x in bad if x["kind"] == kind)[key] = value
        with pytest.raises((ValueError, KeyError)):
            consumer.verify_events(bad, tasks, banks, [2., 1.])


def test_failure_does_not_continue():
    tasks, banks = inputs()
    events = []
    meter = bridge.Meter(events.append, lambda: None, lambda: None)
    with pytest.raises(RuntimeError):
        bridge._execute(tasks, banks, [2., 1.], [Backend(fail=0), Backend()], meter,
                        check_path=lambda *a: None, cost=lambda *a: 0.)
    assert events[-1]["kind"] == "panel_failure"
    assert events[-1]["entry_count"] == 4


def test_exclusive_output_scope(tmp_path):
    with pytest.raises(ValueError):
        host.output_path(str(tmp_path / (host.NAME + "_RUN_x")), "_RUN_")
