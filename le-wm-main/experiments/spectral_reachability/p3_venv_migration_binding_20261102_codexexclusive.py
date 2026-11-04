"""Bind a new P3 development environment; CPU only, no execution authority."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
CHECKPOINT = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
OLD = ROOT / "results/codex_p3_online_budget_host_20261030_v1_RUN_01/prepare.json"
OLD_SHA = "bfab85393221afe02f8ac804130040c4f493d5d3deffdeb5486e14cc6e5cb305"
NAME = "p3_venv_migration_binding_20261102_codexexclusive"
PACKAGES = ("torch", "torchvision", "transformers", "stable_worldmodel",
            "stable_pretraining", "numpy")
LEGACY_RELATIVE = (
    "stable_pretraining/backbone/utils.py", "stable_pretraining/data/dataset_stats.py",
    "stable_worldmodel/wm/lewm/lewm.py", "stable_worldmodel/wm/lewm/module.py",
    "torch/_vmap_internals.py", "torch/autograd/__init__.py", "torch/autograd/functional.py",
    "torchvision/transforms/v2/_container.py", "torchvision/transforms/v2/_geometry.py",
    "torchvision/transforms/v2/_misc.py", "torchvision/transforms/v2/functional/_geometry.py",
    "torchvision/transforms/v2/functional/_misc.py",
    "torchvision/transforms/v2/functional/_type_conversion.py",
    "transformers/models/vit/configuration_vit.py", "transformers/models/vit/modeling_vit.py",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular non-symlink file required: {path}")
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": sha(path)}


def build():
    if Path(sys.prefix).resolve() != VENV.resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("run with the project .venv interpreter")
    old = record(OLD)
    if old["sha256"] != OLD_SHA:
        raise ValueError("old prepare SHA mismatch")
    old_plan = json.loads(OLD.read_text())
    packages = []
    sites = set()
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise ValueError(f"missing module: {name}")
        origin = Path(spec.origin)
        if not origin.resolve().is_relative_to(VENV.resolve()):
            raise ValueError(f"module outside venv: {name}")
        sites.add(str(origin.parent.parent))
        try:
            version = metadata.version(name.replace("_", "-"))
        except metadata.PackageNotFoundError:
            version = None
        packages.append({"module": name, "version": version, "entry": record(origin)})
    if len(sites) != 1:
        raise ValueError("modules must share a declared site-packages directory")
    site = Path(next(iter(sites)))
    sources = [record(site / path) for path in LEGACY_RELATIVE]
    sources += [record(ROOT / "experiments" / "checkpoint_compat.py"),
                record(ROOT / "experiments/spectral_reachability/p3_live_distinct_pair_20261102_codexexclusive.py"),
                record(ROOT / "experiments/spectral_reachability/p3_live_pair_host_20261102_codexexclusive.py"),
                record(ROOT / "experiments/spectral_reachability/p3_live_pair_saved_20261102_codexexclusive.py"),
                record(Path(__file__))]
    checkpoint = [record(CHECKPOINT / name) for name in ("config.json", "weights.pt")]
    executable = Path(sys.executable).resolve()
    return {
        "schema": NAME + ".v1",
        "environment_kind": "new_venv_not_historical",
        "old_new_environment_equivalence": "unknown",
        "python": {"prefix": sys.prefix, "base_prefix": sys.base_prefix,
                   "version": platform.python_version(), "executable": record(executable),
                   "venv_config": record(VENV / "pyvenv.cfg")},
        "packages": packages, "selected_sources": sources,
        "checkpoint": checkpoint, "historical_prepare": old,
        "historical_expected_tensor_hash": old_plan.get("expected_model_hash"),
        "historical_task_sha256": old_plan.get("task_sha256"),
        "historical_bank_sha256": old_plan.get("bank_sha256"),
        "selected_sources_complete_closure": False,
        "dependency_imports_tested": False, "checkpoint_tensor_hash_verified": False,
        "task_bank_rebuilt": False, "numerical_parity_verified": False,
        "host_launcher_compatible": False, "execution_authorized": False,
        "cuda_used": False, "P3_admission": False,
    }


def verify(path, expected_sha):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != expected_sha:
        raise ValueError("external binding SHA required")
    saved = json.loads(path.read_text())
    current = build()
    if saved != current:
        raise ValueError("binding drift")
    return {"status": "new_environment_binding_reconstructed",
            "old_new_environment_equivalence": "unknown",
            "execution_authorized": False, "P3_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--output")
    parser.add_argument("--binding")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.action == "verify":
        print(json.dumps(verify(args.binding, args.sha256), sort_keys=True))
        return
    path = Path(args.output)
    if (not path.is_absolute() or path.parent != ROOT / "results"
            or not path.name.startswith(NAME + "_") or os.path.lexists(path)):
        raise ValueError("new exclusive result path under results required")
    raw = (json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"binding": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "execution_authorized": False}))


if __name__ == "__main__":
    main()
