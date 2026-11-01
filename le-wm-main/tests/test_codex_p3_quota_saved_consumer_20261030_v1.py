import copy
import json
import pytest

from experiments.spectral_reachability import codex_p3_quota_saved_consumer_20261030_v1 as consumer
from experiments.spectral_reachability import codex_p3_quota_native_bridge_20261030_v1 as bridge
from test_codex_p3_quota_native_bridge_20261030_v1 import Backend, inputs


def make_events():
    tasks, banks = inputs()
    events = []
    meter = bridge.Meter(events.append, lambda: None, lambda: None)
    bridge._execute(tasks, banks, [2., 1.], [Backend(), Backend()], meter,
                    check_path=lambda task, raw, path: None,
                    cost=lambda task, path: -path[1][0][0])
    return json.loads(json.dumps(events)), tasks, banks


def test_saved_reconstruction_and_tampering(monkeypatch):
    events, tasks, banks = make_events()
    monkeypatch.setattr(consumer.native, "check_path", lambda task, raw, path: None)
    monkeypatch.setattr(consumer.native, "cost", lambda task, path: -path[1][0][0])
    result = consumer.verify_events(events, tasks, banks, [2., 1.])
    assert result["operation_entries"] == 720
    assert result["scientific_pass"] is False
    for kind, mutate in (
        ("candidate_complete", lambda e: e["row"].update(cost=123)),
        ("candidate_complete", lambda e: e["row"].update(eligible=False)),
        ("resource_return", lambda e: e["entry"].update(resource="predict")),
        ("arm_complete", lambda e: e["winners"].reverse()),
    ):
        changed = copy.deepcopy(events)
        event = next(e for e in changed if e["kind"] == kind)
        mutate(event)
        with pytest.raises(ValueError):
            consumer.verify_events(changed, tasks, banks, [2., 1.])
    with pytest.raises(ValueError):
        consumer.verify_events(events[:-1], tasks, banks, [2., 1.])
