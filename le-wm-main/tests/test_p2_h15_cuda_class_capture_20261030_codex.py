"""CPU fakes only; no torch CUDA, model loader, inference, or environment."""
import copy
import threading
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_cuda_class_capture_20261030_codex as cap
from test_p2_h15_neural_ledger_adapter_20261030_codex import contract
from test_p2_h15_production_manifest_20261030_codex import fake_torch


class Module:
    def __init__(self):
        self._modules = {}
        self.training = False

    def named_modules(self, remove_duplicate=False):
        assert remove_duplicate is False
        def walk(name, module):
            yield name, module
            for key, child in module._modules.items():
                if child is not None:
                    yield from walk(name + "." + key if name else key, child)
        return walk("", self)

    def parameters(self):
        return [SimpleNamespace(requires_grad=False, device="cuda:0", dtype="float32")]

    def buffers(self):
        return [SimpleNamespace(device="cuda:0")]

    def forward(self, *args):
        raise AssertionError("capture must not infer")


@pytest.fixture
def setup(monkeypatch):
    info = cap.pre.preflight()
    model = Module()
    child = Module()
    model._modules = {"first": child, "alias": child}
    torch = fake_torch()
    torch.nn = SimpleNamespace(Module=Module)
    hashes = copy.deepcopy(info["model_hashes"])
    monkeypatch.setattr(cap.prod.signal, "model_hash", lambda _: copy.deepcopy(hashes))
    path = next(p for p in cap.pre.host.historical_inventory() if p.endswith("checkpoint_compat.py"))
    monkeypatch.setattr(cap.prod.inspect, "getfile", lambda cls: path)
    return info, model, torch, hashes


def capture(setup):
    info, model, torch, _ = setup
    return cap.capture(model, torch, "cuda:0", preflight_sha256=core.digest(info))


def test_capture_manifest_roundtrip_and_exclusive_files(setup, tmp_path):
    info, _, _, _ = setup
    value = capture(setup)
    assert [r["name"] for r in value["snapshot"]["classes"]] == ["", "first", "alias"]
    task = cap.pre.host.bridge.build_bridge()["roots"][0]["task"]
    c = contract(task)
    args = (0, c, core.digest(c), core.digest(task), value, core.digest(value), core.digest(info))
    manifest = cap.manifest(*args)
    assert cap.verify_manifest(manifest, core.digest(manifest), *args) == manifest
    assert not manifest["ready_to_execute"] and not manifest["capture_authenticity_verified"]
    path = tmp_path / (cap.NAME + "_capture.json")
    sha = cap.save_new(path, value)
    assert sha == core.digest(value) and cap.read_pinned(path, sha) == value
    with pytest.raises(FileExistsError):
        cap.save_new(path, value)
    with pytest.raises(ValueError, match="file digest"):
        cap.read_pinned(path, "0" * 64)


@pytest.mark.parametrize("change", ["model", "precision", "device", "training", "override", "cycle", "preflight"])
def test_capture_fails_closed(setup, change):
    info, model, torch, hashes = setup
    digest = core.digest(info)
    device = "cuda:0"
    if change == "model":
        hashes["all_buffers"] = "0" * 64
    elif change == "precision":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif change == "device":
        device = "cuda"
    elif change == "training":
        model._modules["first"].training = True
    elif change == "override":
        model.forward = lambda: None
    elif change == "cycle":
        model._modules["loop"] = model
    else:
        digest = "0" * 64
    with pytest.raises(ValueError):
        cap.capture(model, torch, device, preflight_sha256=digest)


def test_thread_rejected_before_touching_model(setup):
    info, model, torch, _ = setup
    errors = []
    def run():
        try:
            cap.capture(model, torch, "cuda:0", preflight_sha256=core.digest(info))
        except ValueError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors == ["main thread required"]


def test_graph_enumeration_mismatch_rejected(setup):
    _, model, _, _ = setup
    model.named_modules = lambda remove_duplicate=False: iter([("", model)])
    with pytest.raises(ValueError, match="registered graph mismatch"):
        capture(setup)


def test_state_changes_during_capture_rejected(setup, monkeypatch):
    _, _, _, hashes = setup
    calls = []
    def changing(_):
        calls.append(1)
        result = copy.deepcopy(hashes)
        if len(calls) > 1:
            result["all_buffers"] = "0" * 64
        return result
    monkeypatch.setattr(cap.prod.signal, "model_hash", changing)
    with pytest.raises(ValueError, match="model/precision mismatch"):
        capture(setup)


@pytest.mark.parametrize("change", ["hash", "class_file", "count", "precision", "source", "root"])
def test_recommitted_capture_and_root_tampering(setup, change):
    info, _, _, _ = setup
    value = capture(setup)
    task = cap.pre.host.bridge.build_bridge()["roots"][0]["task"]
    c = contract(task)
    root = 0
    digest = core.digest(value)
    if change == "hash":
        digest = "0" * 64
    elif change == "class_file":
        value["snapshot"]["classes"][0]["sha256"] = "0" * 64
    elif change == "count":
        value["registered_paths"] += 1
    elif change == "precision":
        value["snapshot"]["precision"]["matmul_TF32"] = True
    elif change == "source":
        value["capture_source_sha256"] = "0" * 64
    else:
        root = 1
    if change != "hash":
        digest = core.digest(value)
    with pytest.raises(ValueError):
        cap.manifest(root, c, core.digest(c), core.digest(task), value, digest, core.digest(info))
