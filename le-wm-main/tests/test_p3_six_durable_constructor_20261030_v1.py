"""CPU-only local journal and constructor ownership failure tests."""
from copy import deepcopy
import os

import pytest

from experiments.spectral_reachability import p3_six_durable_transport_20261030_v1 as journal
from experiments.spectral_reachability import p3_six_owned_constructor_20261030_v1 as owner
from experiments.spectral_reachability import p3_six_native_factory_20261030_v1 as factory


def new_store(tmp_path):
    limits = {k: 1024 * 1024 for k in journal.base.LIMITS}
    limits["physical_png"] = 192 * factory.PNG_CAP
    cap = sum(limits.values()) + 1024 * 1024
    return journal.Store(tmp_path / "owned", limits=limits, cap=cap,
                         reserve=1024 * 1024, floor=1024 * 1024, wall=60)


def audit(store):
    path = store.output / "events.jsonl"
    data = path.read_bytes()
    return journal.inspect(path, journal.base.fsha(data), 1024 * 1024)


class Raw:
    def __init__(self, broken=False):
        self.closed = 0
        self.broken = broken
    def close(self):
        self.closed += 1
        if self.broken:
            raise RuntimeError("close failed")


def test_metered_constructor_and_close_are_durable(tmp_path):
    store = new_store(tmp_path)
    meter = owner.Meter(store, "matched_slot_00")
    raw = Raw()
    adapter = owner.module(construct=lambda meter: raw)
    env = adapter.native_factory(meter)
    env.close()
    inspection = audit(store)
    assert inspection["transport_complete"] and not inspection["full_saved_strict"]
    assert [(e["resource"], e["kind"]) for e in inspection["events"]] == [
        ("native_meter/constructor", "intent"), ("native_meter/constructor", "return"),
        ("native_meter/close", "intent"), ("native_meter/close", "return")]
    assert raw.closed == 1


def test_return_failure_still_closes_even_when_journal_poisoned(tmp_path, monkeypatch):
    store = new_store(tmp_path)
    meter = owner.Meter(store, "matched_slot_00")
    raw = Raw()
    def construct(meter):
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync failure")))
        return raw
    with pytest.raises(OSError, match="fsync failure") as error:
        owner.module(construct=construct).native_factory(meter)
    assert raw.closed == 1 and store.poisoned
    assert any("constructor-owned cleanup failed" in note for note in error.value.__notes__)
    with pytest.raises(ValueError, match="store unavailable"):
        store.event("physical", "record", "unsafe", store.index, {})
    # The return write may be present despite failure; no inference of durability.
    result = audit(store)
    assert not result["physical_certified"]


def test_wrapper_failure_closes_and_cleanup_failure_is_recorded(tmp_path):
    store = new_store(tmp_path)
    meter = owner.Meter(store, "matched_slot_00")
    raw = Raw(broken=True)
    def broken_wrapper(raw, meter):
        raise RuntimeError("wrapper failed")
    with pytest.raises(RuntimeError, match="wrapper failed") as exc:
        owner.module(construct=lambda meter: raw, wrap=broken_wrapper).native_factory(meter)
    assert raw.closed == 1
    assert any("close failed" in note for note in exc.value.__notes__)
    result = audit(store)
    assert result["counts"]["physical:native_meter/close"]["failed"] == 1
    assert not result["transport_complete"]


def test_return_exception_after_committed_return_still_closes(tmp_path, monkeypatch):
    store = new_store(tmp_path)
    meter = owner.Meter(store, "matched_slot_00")
    raw = Raw()
    real_end = meter.end
    def end_then_fail(token):
        real_end(token)
        raise RuntimeError("after return")
    monkeypatch.setattr(meter, "end", end_then_fail)
    with pytest.raises(RuntimeError, match="after return"):
        owner.module(construct=lambda meter: raw).native_factory(meter)
    assert raw.closed == 1
    assert audit(store)["counts"]["physical:native_meter/constructor"]["returned"] == 1


def test_constructor_exception_is_charged_and_not_retried(tmp_path):
    store = new_store(tmp_path)
    meter = owner.Meter(store, "matched_slot_00")
    def broken(meter):
        raise RuntimeError("constructor failed")
    with pytest.raises(RuntimeError):
        owner.module(construct=broken).native_factory(meter)
    counts = audit(store)["counts"]["physical:native_meter/constructor"]
    assert counts == dict(attempted=1, returned=0, failed=1)


def test_partial_and_torn_journal_are_preserved_not_completed(tmp_path):
    store = new_store(tmp_path)
    store.event("physical", "intent", "call", store.index, {})
    path = store.output / "events.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"incomplete":')
    blob = path.read_bytes()
    result = journal.inspect(path, journal.base.fsha(blob), 1024 * 1024)
    assert result["pending"] == {("physical", 0): "call"}
    assert result["trailing_bytes"] == len(b'{"incomplete":')
    assert not result["transport_complete"] and path.read_bytes() == blob
    with pytest.raises(ValueError):
        journal.inspect(path, "0" * 64, 1024 * 1024)


def test_chain_tamper_and_exclusive_output_rejected(tmp_path):
    store = new_store(tmp_path)
    store.event("physical", "record", "event", store.index, {})
    path = store.output / "events.jsonl"
    blob = path.read_bytes().replace(b'"event"', b'"alter"')
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="chain"):
        journal.inspect(path, journal.base.fsha(blob), 1024 * 1024)
    with pytest.raises(ValueError, match="exclusive"):
        new_store(tmp_path)
    with pytest.raises(ValueError, match="publication"):
        store.publish("completed")


def test_factory_injection_and_preflight_without_native_execution(tmp_path):
    store = new_store(tmp_path)
    roots = [dict(root_sha256=c * 64) for c in ("a", "b")]
    prepared = dict(tasks=[dict(binding=dict(root_sha256=x["root_sha256"])) for x in roots])
    source = dict(roots=deepcopy(roots), source_plan={})
    raw = Raw()
    meter_module = owner.module(construct=lambda meter: raw)
    class Reset:
        @staticmethod
        def TracedEnv(env, store, episode):
            return env
    class Native:
        @staticmethod
        def setup(*args):
            raise RuntimeError("synthetic setup failure")
    f = factory.Factory(prepared, source, store, None, None, None,
                        meter_module=meter_module, native_module=Native, reset_module=Reset)
    with pytest.raises(RuntimeError, match="synthetic setup"):
        f(0, 0, "fixed_order")
    assert raw.closed == 1
    result = audit(store)
    assert result["counts"]["physical:native_meter/close"]["returned"] == 1
    assert not result["full_saved_strict"]


def test_byte_cap_rejects_before_write_without_poison(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(ValueError, match="ceiling"):
        store.write("too_large.bin", b"x" * (store.limits["physical_metadata"] + 1), "physical_metadata")
    assert not (store.output / "too_large.bin").exists() and not store.poisoned


def test_source_contract_includes_only_new_adapter_files(monkeypatch):
    monkeypatch.setattr(factory, "source_contract", lambda: {"fixture": True})
    value = owner.source_contract()
    assert len(value["added_sha256"]) == 2 and value["host_runner_ready"] is False
    owner.verify_source_contract(value)
    value["scientific_pass"] = True
    with pytest.raises(ValueError, match="drift"):
        owner.verify_source_contract(value)
