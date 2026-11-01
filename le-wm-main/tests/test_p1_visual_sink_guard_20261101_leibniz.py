import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_visual_sink_guard_20261101_leibniz as adapter

g, visual = adapter.total, adapter.visual


class Handle:
    def __init__(self, module, hook):
        self.module, self.hook = module, hook

    def remove(self):
        self.module.hooks.remove(self.hook)


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self, hook)

    def fire(self, sink, resource, index, *, failure=False):
        for hook in self.hooks:
            hook(self, ())
        event = {"index": index, "resource": resource, "completed": False, "image": 0}
        sink.event({"kind": "module_attempt", "entry": event})
        if failure:
            raise RuntimeError("visual failed")
        sink.event({"kind": "module_return", "entry": {**event, "completed": True}})


class Model:
    def __init__(self):
        for name in (n for names in g.HOOKS.values() for n in names):
            setattr(self, name, Module())


class Store:
    def __init__(self, root):
        self.path, self.events = root, []

    def check(self):
        return None

    def namespace(self, name):
        path = self.path / name
        path.mkdir()
        return path

    def emit(self, kind, **kwargs):
        self.events.append((kind, kwargs))


def make(tmp_path, cap=2):
    sources = [Path(m.__file__).resolve() for m in (adapter, g, visual, adapter.encoding, adapter.moving)]
    contract = {"schema": g.NAME + ".contract.v1", "run_id": "visual-test", "arm": "cem",
                "caps": dict.fromkeys(g.AXES, 10), "identity": dict.fromkeys(g.IDENTITY, "a" * 64),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "shared_initialization": "full_standalone_charge_each_arm", "hook_modules": g.HOOKS,
                "scope": "per_root_per_arm_start_before_load"}
    contract["caps"]["source_encode_calls"] = cap
    guard = g.Guard(tmp_path / "total.jsonl", contract, g.digest(contract))
    model = Model()
    guard.install_model_sentinels(model)
    store = Store(tmp_path)
    meter = visual.Meter(store)
    meter.episode = "test"
    return guard, model, store, meter, contract


def operation(model, store, meter, *, failure=False):
    # Instantiate and exercise the actual legacy Sink.event/begin/end.
    sink = visual.Sink(store, meter, "sample", "encode")
    model.encoder.fire(sink, "encoder", 0, failure=failure)
    model.projector.fire(sink, "projector", 1)
    sink.complete()
    return "latent"


def test_actual_sink_begin_end_paid_once(tmp_path):
    guard, model, store, meter, contract = make(tmp_path)
    assert adapter.guarded_boundary(guard, meter, store, lambda: operation(model, store, meter)) == "latent"
    assert meter.counts["encoder"] == meter.counts["projector"] == {"attempted": 1, "completed": 1}
    result = guard.close()
    audit = g.verify_journal(result["journal"], result["sha256"], g.digest(contract))
    assert audit["charged"]["source_encode_calls"] == audit["charged"]["source_encode_samples"] == 1
    assert not audit["coverage_verified"]


def test_budget_exhausted_before_sink_or_module(tmp_path):
    guard, model, store, meter, contract = make(tmp_path, cap=0)
    with pytest.raises(ValueError, match="budget exceeded"):
        adapter.guarded_boundary(guard, meter, store, lambda: operation(model, store, meter))
    assert not (tmp_path / "sample_encode").exists()
    assert meter.counts["encoder"]["attempted"] == 0
    result = guard.close()
    assert g.verify_journal(result["journal"], result["sha256"], g.digest(contract))["status"] == "failed_or_blocked"


def test_direct_begin_bypass_rejected(tmp_path):
    guard, model, store, meter, contract = make(tmp_path)
    with pytest.raises(ValueError, match="direct meter.begin bypass"):
        adapter.guarded_boundary(guard, meter, store, lambda: meter.begin("encoder"))
    assert meter.counts["encoder"]["attempted"] == 0
    result = guard.close()
    assert g.verify_journal(result["journal"], result["sha256"], g.digest(contract))["status"] == "failed_or_blocked"


def test_sink_cannot_forge_unobserved_module(tmp_path):
    guard, model, store, meter, _ = make(tmp_path)
    def bad():
        sink = visual.Sink(store, meter, "sample", "encode")
        sink.event({"kind": "module_attempt", "entry": {"index": 0, "resource": "encoder"}})
    with pytest.raises(ValueError, match="hook before begin"):
        adapter.guarded_boundary(guard, meter, store, bad)
    assert guard.close()["status"] == "failed_or_blocked"


def test_failed_encoder_keeps_attempt_and_no_refund(tmp_path):
    guard, model, store, meter, contract = make(tmp_path)
    with pytest.raises(RuntimeError, match="visual failed"):
        adapter.guarded_boundary(guard, meter, store, lambda: operation(model, store, meter, failure=True))
    assert meter.counts["encoder"] == {"attempted": 1, "completed": 0}
    assert meter.pending
    result = guard.close()
    audit = g.verify_journal(result["journal"], result["sha256"], g.digest(contract))
    assert audit["charged"]["source_encode_calls"] == 1 and audit["status"] == "failed_or_blocked"


def test_wrong_end_token_fails(tmp_path):
    guard, model, store, meter, _ = make(tmp_path)
    def bad():
        sink = visual.Sink(store, meter, "sample", "encode")
        for hook in model.encoder.hooks:
            hook(model.encoder, ())
        ident = sink.begin("encoder")
        sink.end(ident + 1)
    with pytest.raises(ValueError):
        adapter.guarded_boundary(guard, meter, store, bad)
    assert guard.close()["status"] == "failed_or_blocked"


def test_encode_boundary_actual_signature_and_dispatch(tmp_path, monkeypatch):
    guard, model, store, meter, _ = make(tmp_path)
    def original(handler, runtime, task, layer, row, obs, body, passed_meter, passed_store, step):
        assert (passed_meter, passed_store, step) == (meter, store, 5)
        return operation(model, passed_store, passed_meter)
    monkeypatch.setattr(adapter.moving, "encode_boundary", original)
    assert adapter.encode_boundary(guard, None, {}, {}, {}, {}, {}, None, meter, store, 5) == "latent"
    assert guard.close()["status"] == "closed_unverified"


def test_unbound_visual_source_fails_before_operation(tmp_path):
    guard, model, store, meter, _ = make(tmp_path)
    guard.contract["source_sha256"].pop(str(Path(adapter.encoding.__file__).resolve()))
    with pytest.raises(ValueError, match="visual source binding"):
        adapter.guarded_boundary(guard, meter, store, lambda: operation(model, store, meter))
    assert not (tmp_path / "sample_encode").exists()
    assert guard.close()["status"] == "failed_or_blocked"
