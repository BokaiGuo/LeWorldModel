"""Representation and adapter-guard tests, with CPU-only stand-ins."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_cuda_uuid_bridge_20261030_codex_v2 as b
from experiments.spectral_reachability import p3_decision_native_cuda_adapter_20261030_codex_v1 as a

GPU = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
BYTES = [78, 137, 115, 255, 2, 120, 83, 243, 43, 174, 38, 21, 212, 97, 228, 65]


class CUuuid:
    def __init__(self, data):
        self.bytes = data


def fake_torch():
    state = SimpleNamespace(raw=CUuuid(BYTES.copy()), count=1)
    props = SimpleNamespace(name="GPU fixture")
    def properties(index):
        assert index == 0
        props.uuid = state.raw
        return props
    cuda = SimpleNamespace(device_count=lambda: state.count, get_device_properties=properties,
                           max_memory_allocated=lambda device: 1, max_memory_reserved=lambda device: 2)
    return SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_CUuuid=CUuuid), sentinel=object()), state


def test_host_reported_bytes_wire_order_and_proxy():
    torch, state = fake_torch()
    view = b.TorchView(torch, GPU, GPU, GPU)
    assert view.binding_evidence["bytes_hex"] == "4e8973ff027853f32bae2615d461e441"
    assert view.binding_evidence["physical_uuid"] == GPU
    assert view.cuda.get_device_properties(0).uuid == GPU
    assert view.cuda.get_device_properties(0).name == "GPU fixture"
    assert view.sentinel is torch.sentinel and torch.cuda.get_device_properties(0).uuid is state.raw


@pytest.mark.parametrize("data", [BYTES[:-1], BYTES + [0], [True] + BYTES[1:],
                                  [-1] + BYTES[1:], [256] + BYTES[1:],
                                  BYTES[::-1], "4e8973ff027853f32bae2615d461e441"])
def test_malformed_or_wrong_bytes_rejected(data):
    with pytest.raises(ValueError):
        b.object_evidence(CUuuid(data), CUuuid, GPU)


def test_impostor_object_rejected():
    class Impostor(CUuuid):
        pass
    with pytest.raises(ValueError, match="type"):
        b.object_evidence(Impostor(BYTES), CUuuid, GPU)


@pytest.mark.parametrize("defect", ["smi", "visible", "multiple", "badindex"])
def test_binding_rejects_identity_defects(defect):
    torch, state = fake_torch()
    smi, visible = GPU, GPU
    if defect == "smi":
        smi = "GPU-00000000-0000-0000-0000-000000000000"
    if defect == "visible":
        visible = "0"
    if defect == "multiple":
        state.count = 2
    with pytest.raises(ValueError):
        v = b.TorchView(torch, GPU, smi, visible)
        if defect == "badindex":
            v.cuda.get_device_properties(1)


def test_existing_adapter_guard_checks_every_call_and_poison_on_drift(monkeypatch):
    torch, state = fake_torch()
    view = b.TorchView(torch, GPU, GPU, GPU)
    device = SimpleNamespace(type="cuda", index=0)
    tensor = SimpleNamespace(device=device, requires_grad=False)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(training=False)],
                            named_parameters=lambda: [("p", tensor)], named_buffers=lambda: [])
    obj = object.__new__(a.Adapter)
    obj.poisoned, obj.closed = False, False
    obj.factory = SimpleNamespace(poisoned=False)
    obj.decision_store = SimpleNamespace(poisoned=False, finished=False)
    obj.native_store = SimpleNamespace(check=lambda *args: None)
    obj.torch, obj.device, obj.model, obj.gpu_uuid = view, device, model, GPU
    obj.allocated_cap, obj.reserved_cap, obj.counts = 10, 10, {}
    obj.contract = {"precision": "fixture"}
    monkeypatch.setattr(a.identity.host.prior, "checked_precision", lambda _: "fixture")
    obj.guard("before")
    state.raw = CUuuid([0] + BYTES[1:])
    with pytest.raises(ValueError, match="differ"):
        obj.guard("after")
    assert obj.poisoned and obj.factory.poisoned and obj.counts == {"before": 1}


def test_existing_snapshot_receives_canonical_uuid(monkeypatch):
    torch, _ = fake_torch()
    view = b.TorchView(torch, GPU, GPU, GPU)
    device = SimpleNamespace(type="cuda", index=0)
    tensor = SimpleNamespace(device=device, requires_grad=False)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(training=False)],
                            named_parameters=lambda: [("p", tensor)], named_buffers=lambda: [],
                            parameters=lambda: [tensor])
    monkeypatch.setattr(a.identity.host.source.host.signal, "model_hash",
                        lambda _: {"original_state_dict": "hash"})
    monkeypatch.setattr(a.identity.host.prior, "model_sources", lambda *args: "sources")
    monkeypatch.setattr(a.identity.host.prior, "checked_precision", lambda _: "precision")
    result = a.identity.snapshot(model, view, device, {}, {"expected_model_hash": "hash",
                                                          "precision": "precision"})
    assert result["device"]["uuid"] == GPU
