"""CPU-only integration of the versioned driver; no model or environment."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v2 as d

PLAN = d.ROOT / "results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP02.json"
SHA = "bf54ad72631bd437bb88d90fe71b7065b803901fc25100fa3d93d3fc7b8bed07"
GPU = "GPU-4e8973ff-0278-53f3-2bae-2615d461e441"
RAW = [78, 137, 115, 255, 2, 120, 83, 243, 43, 174, 38, 21, 212, 97, 228, 65]


def test_actual_plan_and_source_closure():
    p = d.read_json(PLAN)
    assert d.old.file_sha(PLAN) == "3e7a8f639b3abc1ba548cb7471df4a244808a76217f2f77f5a60665a66b6afe4"
    assert d.canonical_sha(p) == SHA
    base, prepared = d.preflight(p, SHA)
    assert base["gpu_uuid"] == GPU and len(prepared["tasks"]) == 2


@pytest.mark.parametrize("target", ["driver_v2", "bridge", "legacy_driver", "gpu_preflight",
                                    "checkpoint_compat"])
def test_source_pin_tamper(target):
    p = d.read_json(PLAN)
    p["source_files"][target] = "0" * 64
    with pytest.raises(ValueError, match="v2 source pin"):
        d.source_pins(p)


class CUuuid:
    def __init__(self, raw):
        self.bytes = raw


def setup_runtime(monkeypatch, tmp_path):
    state = SimpleNamespace(raw=CUuuid(RAW.copy()), called=[])
    cuda = SimpleNamespace(device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(uuid=state.raw),
        synchronize=lambda device: state.called.append("synchronize"))
    torch = SimpleNamespace(cuda=cuda, _C=SimpleNamespace(_CUuuid=CUuuid),
        device=lambda s: s, autocast=lambda **kw: nullcontext(),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True)))
    class Model:
        def eval(self):
            state.called.append("eval")
            return self
        def requires_grad_(self, flag):
            assert flag is False
            return self
    def loader(policy, device):
        state.called.append("load")
        return Model()
    def allocation(device):
        state.called.append("allocation")
    class Adapter:
        def __init__(self, **kw):
            state.called.append("adapter")
            assert kw["torch"].cuda.get_device_properties(0).uuid == GPU
            self.session_factory = self.backend_factory = lambda *args: None
        def finish(self):
            state.called.append("finish")
            return {"scientific_pass": False}
    monkeypatch.setattr(d.old.adapter, "Adapter", Adapter)
    monkeypatch.setattr(d.old.control, "execute_roster",
                        lambda tasks, *args: {"schema": "fixture", "rows": list(range(6))})
    monkeypatch.setattr(d, "source_pins", lambda p: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setattr(d.old, "record", lambda root, name, value, plan:
                        state.called.append((name, value)))
    plan = dict(output=str(tmp_path), min_free_mib=6144, min_total_mib=8192)
    base = dict(gpu_uuid=GPU, allocated_cap=3, reserved_cap=4,
                files={"adapter": "a"}, source_sha256="b")
    kwargs = dict(torch=torch, allocation=allocation, loader=loader, sdp_context=nullcontext,
                  snapshot=dict(uuid=GPU, free_mib=7000, total_mib=16000))
    return plan, base, {"policy": "fixture", "tasks": [1, 2]}, state, kwargs


def test_runtime_wires_view_and_saves_raw_identity_before_model(monkeypatch, tmp_path):
    plan, base, prepared, state, kwargs = setup_runtime(monkeypatch, tmp_path)
    receipt, roster, error = d.runtime_with(plan, base, prepared, object(), object(), **kwargs)
    binding = next(x for x in state.called if isinstance(x, tuple))
    assert binding[0] == "uuid_binding_v2.json"
    assert binding[1]["binding"]["bytes_hex"] == "4e8973ff027853f32bae2615d461e441"
    assert state.called.index(binding) < state.called.index("load") < state.called.index("adapter")
    assert receipt["uuid_binding_v2"]["physical_uuid"] == GPU and error is None
    assert len(roster["rows"]) == 6 and "finish" in state.called


@pytest.mark.parametrize("defect", ["bytes", "smi", "visible", "memory"])
def test_runtime_rejects_before_model_or_physics(monkeypatch, tmp_path, defect):
    plan, base, prepared, state, kwargs = setup_runtime(monkeypatch, tmp_path)
    if defect == "bytes":
        state.raw = CUuuid([0] + RAW[1:])
    elif defect == "smi":
        kwargs["snapshot"]["uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
    elif defect == "visible":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    else:
        kwargs["snapshot"]["free_mib"] = 1
    with pytest.raises(ValueError):
        d.runtime_with(plan, base, prepared, object(), object(), **kwargs)
    assert "load" not in state.called and "adapter" not in state.called


def test_smi_uuid_and_mode_checked():
    outputs = iter([GPU + ", 16311, 15000, Default\n", "[N/A]\n"])
    result = d.gpu_snapshot(GPU, runner=lambda *a, **kw: SimpleNamespace(stdout=next(outputs)))
    assert result["uuid"] == GPU and result["free_mib"] == 15000
    outputs = iter(["GPU-00000000-0000-0000-0000-000000000000, 16311, 15000, Default\n", "[N/A]\n"])
    with pytest.raises(ValueError, match="UUID or mode"):
        d.gpu_snapshot(GPU, runner=lambda *a, **kw: SimpleNamespace(stdout=next(outputs)))


def test_v2_run_binds_plan_then_dispatches_via_existing_lifecycle(monkeypatch, tmp_path):
    plan = {"output": str(tmp_path), "base_plan_sha256": "base"}
    base = {"marker": "base"}
    seen = []
    monkeypatch.setattr(d, "preflight", lambda p, h: (base, {}))
    monkeypatch.setattr(d.old, "record", lambda root, name, value, b:
                        seen.append((name, value["plan_sha256"], b)))
    monkeypatch.setattr(d, "host_runtime", lambda p, b, prepared, ds, ns: ("receipt", "roster", None))
    def lifecycle(output, received, pin, auth, *, runtime):
        assert (output, received, pin, auth) == (str(tmp_path), base, "base", d.old.AUTH)
        return runtime(received, {}, object(), object())
    monkeypatch.setattr(d.old, "run", lifecycle)
    assert d.run(plan, "new-plan-pin", d.AUTH) == ("receipt", "roster", None)
    assert seen == [("v2_binding.json", "new-plan-pin", base)]
    with pytest.raises(ValueError, match="authorization"):
        d.run(plan, "new-plan-pin", "wrong")
