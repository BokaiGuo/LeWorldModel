"""Main-thread capture for an already-loaded H15 CUDA model; no loader or inference.

Capture reads CUDA tensor state to CPU and therefore must only be called after
explicit authorization to use the device. Offline verification cannot prove
that a saved declaration came from a genuine runtime or was preregistered.
"""
import copy
import json
import os
import threading
from pathlib import Path

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_h15_preproduction_manifest_20261030_codex as pre
from experiments.spectral_reachability import p2_h15_production_manifest_20261030_codex as prod
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_cuda_class_capture_20261030_codex"
LIMIT = 16 * 1024 * 1024


def _source_sha():
    return spec.sha(Path(__file__).read_bytes())


def _main_thread():
    core.require(threading.current_thread() is threading.main_thread(), "main thread required")


def _graph(model, torch):
    """Enumerate every registered path including aliases, reject registration cycles."""
    rows = []
    def visit(name, module, ancestors):
        core.require(isinstance(module, torch.nn.Module) and id(module) not in ancestors,
                     "invalid module or registration cycle")
        core.require(type(module._modules) is dict, "registered module mapping")
        rows.append((name, module))
        for key, child in module._modules.items():
            core.require(type(key) is str and bool(key) and "." not in key, "module name")
            if child is not None:
                visit(name + "." + key if name else key, child, ancestors | {id(module)})
    visit("", model, set())
    observed = list(model.named_modules(remove_duplicate=False))
    core.require(len(rows) == len(observed) and all(
        a == b and x is y for (a, x), (b, y) in zip(rows, observed)), "registered graph mismatch")
    return rows


def _snapshot(model, torch, device, info, inventory):
    graph = _graph(model, torch)
    core.require(type(device) is str and device.startswith("cuda:") and device[5:].isdigit()
                 and str(int(device[5:])) == device[5:], "explicit CUDA device")
    core.require(all(not module.training and "forward" not in vars(module) for _, module in graph),
                 "eval modules without instance forward overrides")
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    core.require(bool(parameters) and all(not p.requires_grad and str(p.device) == device
                                         and p.dtype == torch.float32 for p in parameters)
                 and all(str(b.device) == device for b in buffers), "frozen CUDA state")
    classes = prod.capture_classes(model)
    core.require([row["name"] for row in classes] == [name for name, _ in graph], "class graph order")
    for row in classes:
        core.require(inventory.get(row["path"]) == row["sha256"], "class source not pinned")
    hashes = prod.signal.model_hash(model)
    precision = prod.native.checked_precision(torch)
    core.require(hashes == info["model_hashes"] and precision == info["precision"], "model/precision mismatch")
    return dict(model_hashes=hashes, classes=classes, precision=precision, device=device)


def capture(model, torch, device, *, preflight_sha256):
    """Capture the supplied live model, without invoking forward or constructing it."""
    _main_thread()
    info = pre.preflight()
    core.require(spec.hash_string(preflight_sha256) and core.digest(info) == preflight_sha256,
                 "external preflight digest")
    files = pre.host.historical_inventory()
    core.require(pre.checked_inventory(files) == info["inventory_sha256"], "inventory before")
    before = _snapshot(model, torch, device, info, files)
    after = _snapshot(model, torch, device, info, files)
    core.require(before == after and pre.checked_inventory(files) == info["inventory_sha256"],
                 "capture drift")
    return dict(schema=NAME + ".capture.v1", capture_source_sha256=_source_sha(),
                preflight_sha256=preflight_sha256, inventory_sha256=info["inventory_sha256"],
                snapshot=before, registered_paths=len(before["classes"]),
                runtime_dependency_closure_verified=False, production_authorized=False)


def _validate_capture(value, expected_sha256, info, preflight_sha256):
    core.require(spec.hash_string(expected_sha256) and core.digest(value) == expected_sha256,
                 "external capture digest")
    core.keys(value, ("schema", "capture_source_sha256", "preflight_sha256", "inventory_sha256",
                      "snapshot", "registered_paths", "runtime_dependency_closure_verified",
                      "production_authorized"))
    core.require(value["schema"] == NAME + ".capture.v1" and value["capture_source_sha256"] == _source_sha()
                 and value["preflight_sha256"] == preflight_sha256
                 and value["inventory_sha256"] == info["inventory_sha256"]
                 and value["runtime_dependency_closure_verified"] is False
                 and value["production_authorized"] is False, "capture bindings")
    snapshot = value["snapshot"]
    core.keys(snapshot, ("model_hashes", "classes", "precision", "device"))
    device = snapshot["device"]
    core.require(type(device) is str and device.startswith("cuda:") and device[5:].isdigit()
                 and str(int(device[5:])) == device[5:]
                 and snapshot["model_hashes"] == info["model_hashes"]
                 and snapshot["precision"] == info["precision"], "capture state")
    rows = snapshot["classes"]
    core.require(type(rows) is list and bool(rows) and type(value["registered_paths"]) is int
                 and len(rows) == value["registered_paths"], "capture class count")
    return dict(model_hashes=snapshot["model_hashes"], classes=rows)


def manifest(root, contract, contract_sha256, task_sha256, capture_value, capture_sha256, preflight_sha256):
    """Build a pinned candidate offline; no claim that capture is authentic."""
    info = pre.preflight()
    core.require(spec.hash_string(preflight_sha256) and core.digest(info) == preflight_sha256,
                 "external preflight digest")
    evidence = _validate_capture(capture_value, capture_sha256, info, preflight_sha256)
    candidate = pre.candidate(root, contract, contract_sha256, task_sha256, evidence, core.digest(evidence))
    return dict(schema=NAME + ".manifest.v1", capture_sha256=capture_sha256,
                capture_source_sha256=_source_sha(), candidate=candidate,
                capture_authenticity_verified=False, runtime_dependency_closure_verified=False,
                ready_to_execute=False, physical_release=False, P1_admission=False, P2_admission=False)


def verify_manifest(value, expected_sha256, root, contract, contract_sha256, task_sha256,
                    capture_value, capture_sha256, preflight_sha256):
    core.require(spec.hash_string(expected_sha256) and core.digest(value) == expected_sha256,
                 "external manifest digest")
    core.require(value == manifest(root, contract, contract_sha256, task_sha256, capture_value,
                                   capture_sha256, preflight_sha256), "manifest reconstruction")
    return copy.deepcopy(value)


def save_new(path, value):
    """Exclusive canonical JSON file; returned digest is over canonical object bytes."""
    path = Path(path)
    core.require(path.is_absolute() and path.name.startswith(NAME + "_")
                 and path.parent.resolve() == path.parent and path.parent.is_dir(), "exclusive output path")
    data = core.canonical(value)
    core.require(len(data) <= LIMIT, "bounded output")
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return spec.sha(data)


def read_pinned(path, expected_sha256):
    path = Path(path)
    core.require(spec.hash_string(expected_sha256) and path.is_absolute() and path.resolve() == path
                 and path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMIT,
                 "bounded pinned file")
    raw = path.read_bytes()
    core.require(spec.sha(raw) == expected_sha256, "file digest")
    def pairs(items):
        result = {}
        for k, v in items:
            core.require(k not in result, "duplicate key")
            result[k] = v
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    core.require(core.canonical(value) == raw, "canonical JSON")
    return value
