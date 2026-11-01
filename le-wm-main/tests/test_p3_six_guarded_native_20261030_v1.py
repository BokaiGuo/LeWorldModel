"""No gym, CUDA, model or physical environment is instantiated."""
from dataclasses import dataclass

import pytest

from experiments.spectral_reachability import p3_six_guarded_native_20261030_v1 as g
import test_p3_six_durable_constructor_20261030_v1 as fixture


@dataclass
class Spec:
    entry_point: object = None


def test_wrapper_failure_cleans_previously_unreturned_base(tmp_path):
    store = fixture.new_store(tmp_path)
    meter = g.original.Meter(store, "slot")
    instances = []
    class Base:
        def __init__(self, **kwargs):
            instances.append(self)
            self.closed = 0
        def close(self):
            self.closed += 1
    def make(spec, **kw):
        spec.entry_point(**kw)
        raise RuntimeError("gym wrapper failure")
    with pytest.raises(RuntimeError, match="gym wrapper"):
        g.owner.module(construct=lambda m: g.tracked_construct(m, make, Spec(), Base)).native_factory(meter)
    assert instances[0].closed == 1
    counts = fixture.audit(store)["counts"]
    assert counts["physical:native_meter/constructor"]["failed"] == 1
    assert counts["physical:native_meter/close"]["returned"] == 1


def test_partial_init_cleanup_is_not_retried(tmp_path):
    store = fixture.new_store(tmp_path)
    meter = g.original.Meter(store, "slot")
    instances = []
    class MeteredBase:
        def __init__(self, **kwargs):
            instances.append(self)
            self.closed = 0
            try:
                raise RuntimeError("partial init")
            except BaseException:
                self.close()
                raise
        def close(self):
            self.closed += 1
    def make(spec, **kw):
        return spec.entry_point(**kw)
    with pytest.raises(RuntimeError, match="partial init"):
        g.owner.module(construct=lambda m: g.tracked_construct(m, make, Spec(), MeteredBase)).native_factory(meter)
    assert instances[0].closed == 1
    assert fixture.audit(store)["counts"]["physical:native_meter/close"]["returned"] == 1


def test_cleanup_failure_remains_failure_and_no_retry(tmp_path):
    store = fixture.new_store(tmp_path)
    meter = g.original.Meter(store, "slot")
    instances = []
    class Base:
        def __init__(self, **kwargs):
            self.closed = 0
            instances.append(self)
        def close(self):
            self.closed += 1
            raise RuntimeError("close broken")
    def make(spec, **kw):
        spec.entry_point(**kw)
        raise RuntimeError("wrapper broken")
    with pytest.raises(RuntimeError, match="wrapper broken") as exc:
        g.owner.module(construct=lambda m: g.tracked_construct(m, make, Spec(), Base)).native_factory(meter)
    assert instances[0].closed == 1
    assert any("cleanup" in note for note in exc.value.__notes__)
    assert fixture.audit(store)["counts"]["physical:native_meter/close"]["failed"] == 1


def test_physical_guard_pre_and_post_with_unconditional_cleanup(tmp_path):
    store = fixture.new_store(tmp_path)
    labels = []
    def guard(label):
        labels.append(label)
        if label == "physical_reset_after":
            raise ValueError("peak")
    module, _ = g.adapters(guard)
    meter = module.Meter(store, "slot")
    called = []
    with pytest.raises(ValueError, match="peak"):
        meter.call("reset", lambda: called.append("reset"))
    meter.cleanup(lambda: called.append("close"))
    assert called == ["reset", "close"]
    assert labels == ["physical_reset_before", "physical_reset_after"]
    assert fixture.audit(store)["pending"]


def test_encoding_guard_before_after_and_exception():
    labels = []
    assert g._around(labels.append, "encoding", lambda: 3) == 3
    assert labels == ["encoding_before", "encoding_after"]
    labels.clear()
    def fail():
        raise RuntimeError("encode failed")
    with pytest.raises(RuntimeError, match="encode failed"):
        g._around(labels.append, "encoding", fail)
    assert labels == ["encoding_before", "encoding_after_failure"]


def test_encoding_preflight_stops_dispatch():
    called = []
    def guard(label):
        raise ValueError("resource limit")
    with pytest.raises(ValueError, match="resource limit"):
        g._around(guard, "encoding", lambda: called.append(True))
    assert not called


def test_successful_handoff_closes_once_through_meter(tmp_path):
    store = fixture.new_store(tmp_path)
    meter = g.original.Meter(store, "slot")
    instances = []
    class Base:
        def __init__(self, **kwargs):
            self.closed = 0
            self.unwrapped = self
            instances.append(self)
        def close(self):
            self.closed += 1
    def make(spec, **kw):
        return spec.entry_point(**kw)
    env = g.owner.module(construct=lambda m: g.tracked_construct(m, make, Spec(), Base)).native_factory(meter)
    env.close()
    assert instances[0].closed == 1
    counts = fixture.audit(store)["counts"]
    assert counts["physical:native_meter/constructor"]["returned"] == 1
    assert counts["physical:native_meter/close"]["returned"] == 1
