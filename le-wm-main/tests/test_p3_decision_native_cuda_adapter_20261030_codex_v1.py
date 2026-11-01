"""CPU stand-ins only: these tests never import torch, run inference, or create an environment."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_decision_native_cuda_adapter_20261030_codex_v1 as m

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def setup(monkeypatch):
    device = SimpleNamespace(type="cuda", index=0)
    class Tensor:
        requires_grad = False
        def __init__(self):
            self.device = device
    class Model:
        def modules(self):
            return [SimpleNamespace(training=False)]
        def named_parameters(self):
            return [("p", Tensor())]
        def named_buffers(self):
            return [("b", Tensor())]
    model = Model()
    cuda = SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid=GPU),
                           max_memory_allocated=lambda d: 50, max_memory_reserved=lambda d: 60)
    torch = SimpleNamespace(cuda=cuda)
    contract = {"precision": {"ok": True}}
    prepared = {"tasks": [{"root": 0}, {"root": 1}]}
    source = {"roots": [{}, {}]}
    snap = {"device": {"uuid": GPU}, "model_hash": "fixture"}
    monkeypatch.setattr(m, "checked_sources", lambda *args: (contract, prepared, source))
    monkeypatch.setattr(m.identity, "snapshot", lambda *args: deepcopy(snap))
    monkeypatch.setattr(m.identity.host, "hook_snapshot", lambda model: ["hooks"])
    monkeypatch.setattr(m.identity.host, "hooks_restored", lambda hooks: hooks == ["hooks"])
    monkeypatch.setattr(m.identity.host.prior, "checked_precision", lambda torch: {"ok": True})
    monkeypatch.setattr(m.control.design.parent, "parent_task",
                        lambda root, context: dict(root=root["root"], root_context=context))
    class DecisionStore:
        finished = poisoned = False
        index, previous = 1, "fixture-tail"
    class NativeStore:
        poisoned = False
        index = 0
        def __init__(self):
            self.events = []
        def check(self, category, size):
            assert category == "numerical_events" and size == 0
        def event(self, origin, kind, resource, token, detail):
            assert token == self.index
            self.events.append(dict(origin=origin, kind=kind, resource=resource, detail=deepcopy(detail)))
            self.index += 1
    class FakeSession:
        episode_id = "fixture-episode"
        def packet(self, start, raw):
            return {"packet": (start, raw)}
        def encode(self, boundary, rgb):
            return {"encode": (boundary, rgb)}
    class Factory:
        def __init__(self, p, s, store, m, t, d, **kwargs):
            assert m is model and t is torch and d is device
            self.active, self.poisoned, self.calls, self.closed = False, False, [], []
        def __call__(self, slot, root, policy):
            self.calls.append((slot, root, policy))
            self.active = True
            def close():
                self.closed.append(slot)
                self.active = False
            return FakeSession(), "prior", close
    class Tree:
        kind = "HOST_CUDA"
        def __init__(self, actual_model, actual_device, task, sink):
            assert actual_model is model and actual_device is device
            self.sink = sink
        def signal(self, request, ledger):
            self.sink.event({"kind": "fake_signal_body"})
            return {"signal": request}
        def replay(self, context, actions, segmented, ledger):
            self.sink.event({"kind": "fake_replay_body"})
            return {"context": context, "actions": actions, "segmented": segmented}
    monkeypatch.setattr(m.decision_journal, "Store", DecisionStore)
    monkeypatch.setattr(m.native_transport, "Store", NativeStore)
    monkeypatch.setattr(m.native_factory, "Factory", Factory)
    monkeypatch.setattr(m.guarded, "adapters", lambda guard: ("meter", "native"))
    monkeypatch.setattr(m.identity.host.prior, "NativeTree", Tree)
    dstore, nstore = DecisionStore(), NativeStore()
    args = dict(model=model, torch=torch, device=device, decision_store=dstore, native_store=nstore,
                gpu_uuid=GPU, allocated_cap=100, reserved_cap=200, authorization=m.AUTH,
                adapter_sha256="a" * 64, source_sha256="b" * 64)
    return args, snap


def test_real_recording_backend_bridge_reuses_sequence_across_decisions(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    session, prior, close = adapter.session_factory(0, 0, "fixed_h1")
    assert prior == "prior" and session.packet(0, "raw") == {"packet": (0, "raw")}
    assert session.encode(5, b"rgb") == {"encode": (5, b"rgb")}
    task = {"root": 0, "root_context": {"actual": "ctx"}}
    one = adapter.backend_factory(task, args["decision_store"])
    two = adapter.backend_factory(task, args["decision_store"])
    assert one.signal({"horizon": 1}, object()) == {"signal": {"horizon": 1}}
    assert two.replay({"x": 1}, [1], False, object())["segmented"] is False
    saved = m.bridge.reconstruct(args["native_store"].events, 0)
    assert [x["call"] for x in saved] == [0, 0, 0, 1, 1, 1]
    assert [x["payload"]["kind"] for x in saved] == [
        "native_call_attempt", "fake_signal_body", "native_call_return",
        "native_call_attempt", "fake_replay_body", "native_call_return"]
    close()
    assert adapter.factory.calls == [(0, 0, "fixed_order")]
    with pytest.raises(ValueError, match="stale"):
        one.signal({"horizon": 1}, object())
    with pytest.raises(ValueError, match="stale"):
        session.packet(0, "raw")
    for slot, root, policy in m.control.ROSTER[1:]:
        _, _, finish = adapter.session_factory(slot, root, policy)
        finish()
    assert adapter.factory.calls == [(s, r, m.POLICY_MAP[p]) for s, r, p in m.control.ROSTER]
    assert adapter.finish()["scientific_pass"] is False


@pytest.mark.parametrize("defect", ["cpu", "authorization", "uuid", "budget"])
def test_invalid_admission_refuses_before_model_or_environment(monkeypatch, defect):
    args, _ = setup(monkeypatch)
    monkeypatch.setattr(m, "checked_sources", lambda *a: pytest.fail("must not enter source path"))
    if defect == "cpu":
        args["device"] = SimpleNamespace(type="cpu", index=0)
    elif defect == "authorization":
        args["authorization"] = "wrong"
    elif defect == "uuid":
        args["gpu_uuid"] = "GPU-fake"
    else:
        args["allocated_cap"] = 0
    with pytest.raises(ValueError):
        m.Adapter(**args)


def test_peak_change_latches_failure_but_cleanup_still_runs(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    session, _, close = adapter.session_factory(0, 0, "fixed_h1")
    args["torch"].cuda.max_memory_reserved = lambda d: 201
    with pytest.raises(ValueError, match="CUDA peak"):
        session.packet(0, "raw")
    assert adapter.poisoned and adapter.factory.poisoned
    close()
    assert adapter.factory.closed == [0]
    with pytest.raises(ValueError):
        adapter.session_factory(1, 0, "signal")
    with pytest.raises(ValueError):
        adapter.finish()


def test_invalid_task_and_sink_do_not_dispatch(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    _, _, close = adapter.session_factory(0, 0, "fixed_h1")
    with pytest.raises(ValueError, match="bound root"):
        adapter.backend_factory({"root": 1, "root_context": {}}, args["decision_store"])
    with pytest.raises(ValueError, match="decision sink"):
        adapter.backend_factory({"root": 0, "root_context": {}}, object())
    close()


def test_backend_error_poison_and_no_retry(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    _, _, close = adapter.session_factory(0, 0, "fixed_h1")
    backend = adapter.backend_factory({"root": 0, "root_context": {}}, args["decision_store"])
    monkeypatch.setattr(backend.backend.backend, "signal", lambda *a: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError):
        backend.signal({}, object())
    assert adapter.poisoned and adapter.factory.poisoned
    with pytest.raises(ValueError, match="poisoned"):
        backend.signal({}, object())
    close()


def test_bridge_setup_failure_closes_native_session(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    def fail(*args):
        raise RuntimeError("transport")
    monkeypatch.setattr(m.bridge, "Bridge", fail)
    with pytest.raises(RuntimeError, match="transport"):
        adapter.session_factory(0, 0, "fixed_h1")
    assert adapter.factory.closed == [0] and adapter.active_slot is None
    assert adapter.poisoned and adapter.factory.poisoned


def test_cleanup_runs_before_failed_anchor_and_latches(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    _, _, close = adapter.session_factory(0, 0, "fixed_h1")
    def fail(*args):
        raise OSError("journal failure")
    monkeypatch.setattr(args["native_store"], "event", fail)
    with pytest.raises(OSError, match="journal failure"):
        close()
    assert adapter.factory.closed == [0] and adapter.active_slot is None
    assert adapter.poisoned and adapter.factory.poisoned


def test_precision_or_uuid_drift_prevents_packet(monkeypatch):
    args, _ = setup(monkeypatch)
    adapter = m.Adapter(**args)
    session, _, close = adapter.session_factory(0, 0, "fixed_h1")
    args["torch"].cuda.get_device_properties = lambda i: SimpleNamespace(uuid="GPU-other")
    with pytest.raises(ValueError, match="UUID drift"):
        session.packet(0, "must not dispatch")
    assert adapter.poisoned
    close()
