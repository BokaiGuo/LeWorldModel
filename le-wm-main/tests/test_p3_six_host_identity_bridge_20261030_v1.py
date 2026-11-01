"""CPU substitutes only: no model loading, CUDA, or physical environment."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_six_host_identity_bridge_20261030_v1 as m


class Device:
    type = "cuda"
    index = 0

    def __str__(self):
        return "cuda:0"


class Model:
    training = False

    def __init__(self, device):
        self.tensor = SimpleNamespace(device=device, requires_grad=False)
        self._forward_hooks = {}
        self._forward_pre_hooks = {}
        self._backward_hooks = {}

    def modules(self):
        return [self]

    def parameters(self):
        return [self.tensor]

    def named_parameters(self):
        return [("weight", self.tensor)]

    def named_buffers(self):
        return []


def setup(monkeypatch):
    device = Device()
    model = Model(device)
    props = SimpleNamespace(uuid="GPU-fixture", name="fixture")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda index: props))
    contract = dict(expected_model_hash="a" * 64, precision={"fixture": True})
    prepared = {"sources": {}}
    monkeypatch.setattr(m.host.source.host.signal, "model_hash",
                        lambda model: dict(original_state_dict="a" * 64, all_buffers="b" * 64))
    monkeypatch.setattr(m.host.prior, "model_sources", lambda model, prepared: ["fixture"])
    monkeypatch.setattr(m.host.prior, "checked_precision", lambda torch: {"fixture": True})
    monkeypatch.setattr(m.producer.strict, "source_material", lambda: (contract.copy(), prepared, {}))
    store = object()
    factory = object.__new__(m.producer.native_factory.Factory)
    factory.store, factory.model, factory.torch, factory.device = store, model, torch, device
    return store, factory, props, contract, prepared


def run(store, factory, contract):
    return m.execute(store, factory, None, source_sha256=m.producer.strict.core.digest(contract),
                     producer_sha256=m.producer.own_sha(), bridge_sha256=m.own_sha())


def test_bound_before_after_and_terminal(monkeypatch):
    store, factory, props, contract, _ = setup(monkeypatch)
    monkeypatch.setattr(m.producer, "execute", lambda *a, **k: {"terminal": {"report_sha256": "c"}})
    receipt = run(store, factory, contract)
    assert receipt["before"] == receipt["after"]
    assert receipt["before"]["device"]["uuid"] == "GPU-fixture"
    assert receipt["terminal"]["report_sha256"] == "c"
    assert not receipt["physical_certified"] and not receipt["full_saved_strict"]


@pytest.mark.parametrize("defect", ["uuid", "weights", "precision", "device", "training", "gradient"])
def test_preflight_refuses_missing_or_mismatched_identity(monkeypatch, defect):
    store, factory, props, contract, _ = setup(monkeypatch)
    if defect == "uuid":
        props.uuid = None
    elif defect == "weights":
        contract["expected_model_hash"] = "0" * 64
    elif defect == "precision":
        contract["precision"] = {}
    elif defect == "device":
        factory.model.tensor.device = object()
    elif defect == "training":
        factory.model.training = True
    else:
        factory.model.tensor.requires_grad = True
    monkeypatch.setattr(m.producer, "execute", lambda *a, **k: pytest.fail("producer started"))
    with pytest.raises(ValueError):
        run(store, factory, contract)


@pytest.mark.parametrize("defect", ["uuid", "hook", "weight", "source"])
def test_postflight_refuses_drift(monkeypatch, defect):
    store, factory, props, contract, _ = setup(monkeypatch)
    def execute(*args, **kwargs):
        if defect == "uuid":
            props.uuid = "different"
        elif defect == "hook":
            factory.model._forward_hooks["extra"] = object()
        elif defect == "weight":
            monkeypatch.setattr(m.host.source.host.signal, "model_hash",
                                lambda model: dict(original_state_dict="0" * 64, all_buffers="b" * 64))
        else:
            monkeypatch.setattr(m.producer.strict, "source_material",
                                lambda: (dict(contract, changed=True), {}, {}))
        return {"terminal": {}}
    monkeypatch.setattr(m.producer, "execute", execute)
    with pytest.raises(ValueError):
        run(store, factory, contract)


def test_producer_failure_propagates_and_postflight_runs(monkeypatch):
    store, factory, _, contract, _ = setup(monkeypatch)
    checks = []
    monkeypatch.setattr(m.host.prior, "checked_precision",
                        lambda torch: checks.append(True) or {"fixture": True})
    def fail(*args, **kwargs):
        raise RuntimeError("producer failure")
    monkeypatch.setattr(m.producer, "execute", fail)
    with pytest.raises(RuntimeError, match="producer failure"):
        run(store, factory, contract)
    assert len(checks) == 2
