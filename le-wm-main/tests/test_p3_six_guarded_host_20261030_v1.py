"""Host orchestration tested with CPU-only substituted runtime objects."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_six_guarded_host_20261030_v1 as h
from experiments.spectral_reachability import p3_six_guarded_host_saved_20261030_v1 as saved
import test_p3_six_host_driver_20261030_v1 as old_fixture


def prepare(monkeypatch):
    args = old_fixture.setup(monkeypatch)
    class Store:
        def __init__(self, path, **kwargs):
            self.output = path
            path.mkdir()
            self.clock, self.started, self.wall = lambda: 1, 0, kwargs["wall"]
            self.checked = []
        def check(self, category, size):
            self.checked.append(category)
    class Factory:
        def __init__(self, prepared, data, store, model, torch, device):
            self.store, self.model, self.torch, self.device = store, model, torch, device
            self.next_slot, self.active, self.poisoned = 0, False, False
    monkeypatch.setattr(h.producer, "BoundStore", Store)
    monkeypatch.setattr(h.producer.native_factory, "Factory", Factory)
    plan, model, torch, device, oldpins = args
    pins = dict(**oldpins, native_sha256=h.file_sha(h.native), host_sha256=h.file_sha(h))
    return plan, model, torch, device, pins


def test_host_installs_native_adapters_and_guards(tmp_path, monkeypatch):
    plan, model, torch, device, pins = prepare(monkeypatch)
    def execute(store, factory, backend, **kw):
        assert factory.meter_module.Meter is not h.native.original.Meter
        assert factory.native_module.Session is not h.native.factory.native.Session
        backend({}, object()).check()
        snap = h.identity.snapshot(model, torch, device, {}, {})
        return dict(before=snap, after=snap, terminal={"report_sha256": "c"*64})
    monkeypatch.setattr(h.identity, "execute", execute)
    out = h.run(tmp_path / "outer", model, torch, device, plan, **pins)
    end = h.base.decode((tmp_path / "outer" / "end.json").read_bytes())
    assert end["guard_counts"]["numerical_call"] == 1
    assert out["scientific_pass"] is False


def test_guard_fault_poisoned_before_next_slot(tmp_path, monkeypatch):
    plan, model, torch, device, pins = prepare(monkeypatch)
    def execute(store, factory, backend, **kw):
        torch.cuda.max_memory_reserved = lambda device: 201
        with pytest.raises(ValueError, match="CUDA peak"):
            backend({}, object())
        assert factory.poisoned
        return {}
    monkeypatch.setattr(h.identity, "execute", execute)
    with pytest.raises(ValueError, match="cleanup"):
        h.run(tmp_path / "outer", model, torch, device, plan, **pins)
    assert (tmp_path / "outer" / "failure.json").is_file()


@pytest.mark.parametrize("defect", [None, "guards", "receipt", "peaks", "pin"])
def test_saved_join_rejects_rehashed_tampering(tmp_path, monkeypatch, defect):
    plan, model, torch, device, pins = prepare(monkeypatch)
    source = pins["source_sha256"]
    def execute(store, factory, backend, **kw):
        snap = h.identity.snapshot(model, torch, device, {}, {})
        return dict(schema=h.identity.NAME + ".receipt", before=snap, after=snap,
                    source_contract_sha256=source, producer_sha256=pins["producer_sha256"],
                    bridge_sha256=pins["bridge_sha256"], identity_is_attestation=False,
                    physical_certified=False, full_saved_strict=False, scientific_pass=False,
                    terminal=dict(report_sha256="c"*64, completion_sha256="d"*64))
    monkeypatch.setattr(h.identity, "execute", execute)
    root = tmp_path / "outer"
    out = h.run(root, model, torch, device, plan, **pins)
    endpath = root / "end.json"
    end = h.base.decode(endpath.read_bytes())
    end["guard_counts"].update(physical_constructor_before=6, physical_constructor_after=6,
                               encoding_before=24, encoding_after=24, numerical_call=1)
    if defect == "guards":
        end["guard_counts"]["encoding_after"] = 23
    if defect == "receipt":
        end["receipt"]["after"]["device"]["uuid"] = "other"
    if defect == "peaks":
        end["peaks"]["reserved"] = 201
    raw = h.base.raw(end)
    endpath.write_bytes(raw)
    monkeypatch.setattr(h.producer.strict.native_saved, "_pinned",
                        lambda *a: {k: plan[k] for k in ("limits", "cap", "reserve", "floor", "wall")})
    monkeypatch.setattr(h.producer, "consume", lambda *a: {"fixture": True})
    kwargs = dict(begin_sha256=out["begin_sha256"], end_sha256=h.base.fsha(raw),
                  report_sha256="c"*64, completion_sha256="d"*64,
                  **{k: v for k, v in pins.items() if k != "authorization"})
    if defect == "pin":
        kwargs["native_sha256"] = "0"*64
    if defect is None:
        assert saved.consume(root, **kwargs)["guarded_identity_joined"]
    else:
        with pytest.raises(ValueError):
            saved.consume(root, **kwargs)
