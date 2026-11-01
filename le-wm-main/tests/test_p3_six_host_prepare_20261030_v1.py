"""CPU-only callback fixtures; no HOST or physical execution."""
from copy import deepcopy

import pytest

import test_p3_feedback_reverse_control_20261030_v1 as fixture
from experiments.spectral_reachability import p3_six_saved_consumer_20261030_v1 as saved
from experiments.spectral_reachability import p3_six_host_prepare_20261030_v1 as prepare
from experiments.spectral_reachability import p3_six_numeric_bridge_20261030_v1 as bridge
from experiments.spectral_reachability import p3_six_lifecycle_20261030_v1 as lifecycle
import agent_p3_fullcommit_live_20261028_v1_tests as base_fixture


@pytest.mark.parametrize("policy", fixture.m.POLICIES)
def test_complete_callback_records(policy):
    result, events = fixture.run(policy)
    audit = saved.audit_slot(result, events)
    assert audit["attempted"] == {"packet": 4, "encode": 4}
    assert audit["physical_steps"] is None
    assert not audit["full_saved_strict"] and not audit["scientific_pass"]


def test_failed_callback_accounting_preserved():
    class Broken(fixture.Session):
        def encode(self, end, rgb):
            raise RuntimeError("failure")
    result, events = fixture.run("reverse_signal_order", Broken())
    audit = saved.audit_slot(result, events)
    assert audit["status"] == "failed"
    assert audit["attempted"] == {"packet": 1, "encode": 1}
    assert audit["completed"] == {"packet": 1}


def test_tampered_and_missing_transitions_rejected():
    result, events = fixture.run("reverse_signal_order")
    changed = deepcopy(events)
    changed.pop(next(i for i, x in enumerate(changed) if x["kind"] == "callback_attempt"))
    with pytest.raises(ValueError):
        saved.audit_slot(result, changed)
    changed = deepcopy(events)
    changed.pop(next(i for i, x in enumerate(changed) if x["kind"] == "resource_return"))
    with pytest.raises(ValueError):
        saved.audit_slot(result, changed)
    changed = deepcopy(result)
    changed["blocks"][0]["horizon"] = 4 - changed["blocks"][0]["horizon"]
    events = deepcopy(events)
    events[-1]["result"] = changed
    with pytest.raises(ValueError):
        saved.audit_slot(changed, events)


def test_missing_slots_cannot_be_promoted():
    rows = [dict(slot=s, root=r, policy=p, status="not_started", result=None)
            for s, r, p in fixture.m.ROSTER]
    result, events = fixture.run("fixed_order")
    rows[0].update(status=result["status"], result=result)
    report = saved.consume(rows, {0: events})
    assert report["slots"][1:] == [None] * 5
    assert not report["physical_certified"]
    rows[1]["status"] = "callback_feedback_complete_not_certified"
    with pytest.raises(ValueError):
        saved.consume(rows, {0: events})


def test_prepare_declares_six_slot_limits_unadmitted(monkeypatch):
    monkeypatch.setattr(prepare.old_host, "build", lambda: {"implementation_sha256": {}})
    value = prepare.build()
    assert len(value["roster"]) == 6
    assert value["frame_count_six_slots"] == 192
    assert value["six_slot_storage_cap_bytes"] is None
    assert not value["storage_admitted"] and not value["host_runner_ready"]
    assert not value["preregistered"] and not value["scientific_pass"]
    prepare.verify(value)
    value["callback_caps"]["packet"] = 5
    with pytest.raises(ValueError):
        prepare.verify(value)


def test_six_slot_chunk_roundtrip_and_incomplete_rejection():
    class Store:
        def __init__(self):
            self.events = []
            self.index = 0

        def event(self, origin, kind, resource, token, detail):
            self.events.append(dict(origin=origin, kind=kind, resource=resource, detail=detail))
            self.index += 1

    store = Store()
    writer = bridge.Bridge(store, 5)
    payload = {"text": "x" * 13000}
    writer.event(payload)
    assert bridge.reconstruct(store.events, 5) == [payload]
    with pytest.raises(ValueError):
        bridge.reconstruct(store.events[:-1], 5)
    changed = deepcopy(store.events)
    changed[0]["detail"]["data"] = "!"
    with pytest.raises(ValueError):
        bridge.reconstruct(changed, 5)
    with pytest.raises(ValueError):
        bridge.Bridge(store, 6)


def test_lifecycle_setup_failure_cleanup_and_six_slot_records():
    class Store:
        def __init__(self):
            self.events = []
            self.index = 0
        def event(self, origin, kind, resource, token, detail):
            self.events.append(dict(origin=origin, kind=kind, resource=resource, detail=detail))
            self.index += 1
    root, raw = fixture.setup()
    closed = []
    def create(slot, root_index, policy):
        if slot == 2:
            raise RuntimeError("setup failure")
        return fixture.Session(), raw, lambda: closed.append(slot)
    store = Store()
    result = lifecycle.execute([root, root], store, create,
                               lambda task, sink: base_fixture.RamBackend(task, base_fixture.Events()),
                               permit_ram=True)
    assert len(result["rows"]) == 6 and result["rows"][2]["status"] == "engineering_failed"
    assert closed == [0, 1, 3, 4, 5]
    report = saved.consume_transport(store.events)
    assert report["slots"][2] is None and not report["physical_certified"]
    for slot in closed:
        wrapped = bridge.reconstruct(store.events, slot)
        assert all(item["slot"] == slot and item["kind"] == "arm_event" for item in wrapped)
        payloads = [item["payload"] for item in wrapped]
        final = payloads[-1]["result"]
        assert fixture.m.core.digest(final) == result["rows"][slot]["result_sha256"]
        assert saved.audit_slot(final, payloads)["status"] == "callback_feedback_complete_not_certified", final["errors"]
