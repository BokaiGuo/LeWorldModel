"""All runtime and backend objects here are CPU-only stand-ins."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_six_host_driver_20261030_v1 as d
from experiments.spectral_reachability import p3_six_host_driver_saved_20261030_v1 as saved


def setup(monkeypatch):
    plan = dict(limits={k: 1024 for k in d.base.LIMITS}, cap=20000, reserve=1000,
                floor=1, wall=30, outer_cap=3*d.MAX_RECORD,
                cuda_allocated_bytes=100, cuda_reserved_bytes=200, coordination="main-test-only")
    model, device = object(), SimpleNamespace(index=0)
    cuda = SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid="fixture"),
                           max_memory_allocated=lambda dev: 50, max_memory_reserved=lambda dev: 60)
    torch = SimpleNamespace(cuda=cuda)
    snap = dict(model_hash={"original_state_dict": "a"*64}, precision={"ok": True},
                device={"uuid": "fixture"})
    contract = dict(expected_model_hash="a"*64, precision={"ok": True})
    monkeypatch.setattr(d.identity, "snapshot", lambda *args: deepcopy(snap))
    monkeypatch.setattr(d.producer.strict, "source_material", lambda: (contract, {}, {}))
    monkeypatch.setattr(d.identity.host.prior, "checked_precision", lambda t: {"ok": True})
    class Store:
        def __init__(self, path, **kw):
            self.output = path
            path.mkdir()
        def check(self, category, size):
            return None
    class Factory:
        def __init__(self, prepared, data, store, m, t, dev):
            self.model, self.device = m, dev
            self.active = self.poisoned = False
    class Tree:
        def __init__(self, m, dev, task, sink):
            assert m is model and dev is device
    class Backend:
        def __init__(self, tree, task, sink, check):
            self.check = check
    monkeypatch.setattr(d.producer, "BoundStore", Store)
    monkeypatch.setattr(d.producer.native_factory, "Factory", Factory)
    monkeypatch.setattr(d.identity.host.prior, "NativeTree", Tree)
    monkeypatch.setattr(d.identity.host.prior, "RecordingBackend", Backend)
    source = d.producer.strict.core.digest(contract)
    pins = dict(authorization=d.AUTH, plan_sha256=d.base.digest(plan), source_sha256=source,
                producer_sha256=d.producer.own_sha(), bridge_sha256=d.identity.own_sha(),
                driver_sha256=d.own_sha())
    def execute(store, factory, backend, **kwargs):
        backend({}, object()).check()
        return dict(schema=d.identity.NAME + ".receipt", before=deepcopy(snap), after=deepcopy(snap),
                    bridge_sha256=pins["bridge_sha256"], source_contract_sha256=source,
                    producer_sha256=pins["producer_sha256"], terminal=dict(report_sha256="c"*64,
                    completion_sha256="d"*64), identity_is_attestation=False, full_saved_strict=False,
                    physical_certified=False, scientific_pass=False)
    monkeypatch.setattr(d.identity, "execute", execute)
    return plan, model, torch, device, pins


def run(path, args):
    plan, model, torch, device, pins = args
    return d.run(path, model, torch, device, plan, **pins)


def test_driver_records_and_independent_join(tmp_path, monkeypatch):
    args = setup(monkeypatch)
    root = tmp_path / "out"
    out = run(root, args)
    plan, _, _, _, pins = args
    report = {k: plan[k] for k in ("limits", "cap", "reserve", "floor", "wall")}
    monkeypatch.setattr(d.producer.strict.native_saved, "_pinned", lambda *a: report)
    monkeypatch.setattr(d.producer, "consume", lambda *a: {"stub": True})
    kwargs = dict(begin_sha256=out["begin_sha256"], end_sha256=out["end_sha256"],
                  plan_sha256=pins["plan_sha256"], source_sha256=pins["source_sha256"],
                  producer_sha256=pins["producer_sha256"], bridge_sha256=pins["bridge_sha256"],
                  driver_sha256=pins["driver_sha256"], report_sha256="c"*64, completion_sha256="d"*64)
    result = saved.consume(root, **kwargs)
    assert result["outer_identity_joined"] and not result["physical_certified"]
    (root / "end.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA mismatch"):
        saved.consume(root, **kwargs)


@pytest.mark.parametrize("defect", ["identity", "terminal", "peaks", "begin", "guard"])
def test_rehashed_tampering_is_rejected(tmp_path, monkeypatch, defect):
    args = setup(monkeypatch)
    root = tmp_path / "out"
    out = run(root, args)
    plan, _, _, _, pins = args
    report = {k: plan[k] for k in ("limits", "cap", "reserve", "floor", "wall")}
    monkeypatch.setattr(d.producer.strict.native_saved, "_pinned", lambda *a: report)
    monkeypatch.setattr(d.producer, "consume", lambda *a: pytest.fail("should reject first"))
    path = root / "end.json"
    end = d.base.decode(path.read_bytes())
    if defect == "identity":
        end["receipt"]["after"]["device"]["uuid"] = "other"
    elif defect == "terminal":
        end["receipt"]["terminal"]["report_sha256"] = "0"*64
    elif defect == "peaks":
        end["peaks"]["reserved"] = 201
    elif defect == "begin":
        end["begin_sha256"] = "0"*64
    else:
        end["guard_calls"] = 0
    raw = d.base.raw(end)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        saved.consume(root, begin_sha256=out["begin_sha256"], end_sha256=d.base.fsha(raw),
                      plan_sha256=pins["plan_sha256"], source_sha256=pins["source_sha256"],
                      producer_sha256=pins["producer_sha256"], bridge_sha256=pins["bridge_sha256"],
                      driver_sha256=pins["driver_sha256"], report_sha256="c"*64, completion_sha256="d"*64)


@pytest.mark.parametrize("defect", ["authorization", "plan", "budget"])
def test_preflight_failure_never_starts(tmp_path, monkeypatch, defect):
    args = setup(monkeypatch)
    plan, _, _, _, pins = args
    if defect == "authorization":
        pins["authorization"] = "no"
    elif defect == "plan":
        pins["plan_sha256"] = "0" * 64
    else:
        plan["cuda_reserved_bytes"] = 0
    root = tmp_path / "out"
    with pytest.raises(ValueError):
        run(root, args)
    assert not root.exists()


def test_runtime_peak_failure_is_durable_and_no_end(tmp_path, monkeypatch):
    args = setup(monkeypatch)
    args[2].cuda.max_memory_allocated = lambda device: 101
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="CUDA peak"):
        run(root, args)
    assert (root / "begin.json").is_file() and (root / "failure.json").is_file()
    assert not (root / "end.json").exists()


def test_backend_failure_retained(tmp_path, monkeypatch):
    args = setup(monkeypatch)
    def fail(*a, **kw):
        raise RuntimeError("backend failed")
    monkeypatch.setattr(d.identity, "execute", fail)
    root = tmp_path / "out"
    with pytest.raises(RuntimeError, match="backend failed"):
        run(root, args)
    assert b"backend failed" in (root / "failure.json").read_bytes()


def test_guard_latches_failure_when_lifecycle_swallows_exception(tmp_path, monkeypatch):
    args = setup(monkeypatch)
    torch = args[2]
    def execute(store, factory, backend, **kwargs):
        torch.cuda.max_memory_allocated = lambda device: 101
        with pytest.raises(ValueError, match="CUDA peak"):
            backend({}, object())
        assert factory.poisoned
        return {"before": {}, "after": {}, "terminal": {}}
    monkeypatch.setattr(d.identity, "execute", execute)
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="cleanup"):
        run(root, args)
    assert (root / "failure.json").exists()


def test_refuses_unpinned_driver_or_nonexclusive_output(tmp_path, monkeypatch):
    args = setup(monkeypatch)
    args[-1]["driver_sha256"] = "0"*64
    with pytest.raises(ValueError, match="implementation pins"):
        run(tmp_path / "out", args)
    args[-1]["driver_sha256"] = d.own_sha()
    with pytest.raises(ValueError, match="exclusive"):
        run(tmp_path, args)
