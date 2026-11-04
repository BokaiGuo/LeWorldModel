"""Explicit execution-dependency seal for future v3 research runs.

This is a declared minimum set, not an assertion of a closed environment.
Call capture before planning and verify against the result after execution.
"""
import hashlib
import importlib.util
from pathlib import Path


MODULES = (
    "experiments.spectral_reachability.pc_response_temporal_feedback_v3",
    "experiments.spectral_reachability.pc_response_temporal_feedback_v1",
    "experiments.spectral_reachability.pc_response_temporal_search_v1",
    "experiments.spectral_reachability.pc_response_temporal_warm_cem_v1",
    "experiments.spectral_reachability.pc_response_temporal_v3_reacher_adapter",
    "experiments.pc_budget_allocation_20261103_codex",
    "experiments.pc_equal_resource_dev_20261103_codex",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(extra_paths=()):
    files = {}
    for module in MODULES:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise FileNotFoundError(f"missing dependency {module}")
        path = Path(spec.origin).resolve(strict=True)
        files[str(path)] = digest(path)
    for item in extra_paths:
        path = Path(item).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"not a file: {path}")
        files[str(path)] = digest(path)
    return {"schema": "pc_v3_declared_dependency_seal_v1", "modules": list(MODULES),
            "files": dict(sorted(files.items())), "environment_closure_complete": False}


def verify(seal):
    if seal.get("schema") != "pc_v3_declared_dependency_seal_v1" or seal.get("modules") != list(MODULES):
        raise ValueError("dependency seal schema or declared module set mismatch")
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("missing dependency hashes")
    expected_modules = capture()["files"]
    if not set(expected_modules).issubset(files):
        raise ValueError("missing required module")
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"dependency changed: {path}")
