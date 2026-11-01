#!/usr/bin/env python3
"""Read-only static inventory for H2; never imports the driver or initializes CUDA."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat


PROJECT = Path(__file__).resolve().parents[2]
ENTRY = "experiments.spectral_reachability.codex_p1_blind_h2_host_cuda_20261030_v1"
MODEL = Path("/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm")
BUNDLE = PROJECT / "results/codex_p1_blind_bundle_export_20261030_v1_MAIN01.json"
BUNDLE_SHA = "b5748d81de35df501cb6b7d2d325642080994701a9fa7a634e50ac1c22462856"
LIBRARIES = ["/usr/lib/x86_64-linux-gnu/libcuda.so.1",
             "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"]
DEVICES = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm",
           "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module_path(root, name):
    if name != "experiments" and not name.startswith("experiments."):
        return None
    base = root.joinpath(*name.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root / "experiments"):
            return p
    return None


def source_census(root, entry):
    pending, seen, external, dynamic = [entry], {}, set(), []
    while pending:
        name = pending.pop()
        path = module_path(root, name)
        if path is None or str(path) in seen:
            continue
        data = path.read_bytes()
        tree = ast.parse(data, filename=str(path))
        seen[str(path)] = hashlib.sha256(data).hexdigest()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    dynamic.append({"file": str(path), "line": node.lineno, "kind": "relative_import"})
                    continue
                prefix = node.module or ""
                names = [prefix] + [prefix + "." + a.name for a in node.names if a.name != "*"]
            else:
                names = []
                if isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name) and node.func.id == "__import__" or
                    isinstance(node.func, ast.Attribute) and node.func.attr in ("import_module", "exec_module")
                ):
                    dynamic.append({"file": str(path), "line": node.lineno, "kind": "dynamic_import"})
            for imported in names:
                if imported == "experiments" or imported.startswith("experiments."):
                    if module_path(root, imported) is not None:
                        pending.append(imported)
                elif imported:
                    external.add(imported.split(".")[0])
        # Include concrete package initializers used in this import path.
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
    return {"files_sha256": dict(sorted(seen.items())), "external_top_level_candidates": sorted(external),
            "dynamic_or_relative_imports": sorted(dynamic, key=lambda x: (x["file"], x["line"]))}


def metadata(path):
    p = Path(path)
    try:
        info = p.lstat()
        result = {"path": path, "present": True, "symlink": stat.S_ISLNK(info.st_mode),
                  "resolved": str(p.resolve(strict=True)), "mode": oct(info.st_mode & 0o7777)}
        target = p.stat()
        result.update(character_device=stat.S_ISCHR(target.st_mode),
                      regular_file=stat.S_ISREG(target.st_mode), size=target.st_size)
        if stat.S_ISCHR(target.st_mode):
            result["major_minor"] = [os.major(target.st_rdev), os.minor(target.st_rdev)]
        return result
    except OSError as exc:
        return {"path": path, "present": False, "error": str(exc)}


def inventory():
    source = source_census(PROJECT, ENTRY)
    bundle_sha = digest(BUNDLE)
    if bundle_sha != BUNDLE_SHA:
        raise ValueError("blind bundle hash mismatch")
    bundle = json.loads(BUNDLE.read_bytes())
    expected = bundle["model_files"]
    if set(expected) != {"weights.pt", "config.json"}:
        raise ValueError("unexpected model file inventory")
    model = {}
    for name in sorted(expected):
        path = MODEL / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("model file missing or symlink")
        actual = digest(path)
        if actual != expected[name]:
            raise ValueError("model hash mismatch")
        model[str(path)] = actual
    return {
        "schema": "exploratory-h2-static-dependency-inventory-v1",
        "production_authorized": False, "complete_dependency_census": False,
        "scope": "current execution environment only; host CUDA not tested",
        "entry": ENTRY, "source": source,
        "bundle": {"path": str(BUNDLE), "sha256": bundle_sha},
        "model_files_sha256": model,
        "python_venv": metadata(str(PROJECT / ".venv/bin/python")),
        "venv_config": metadata(str(PROJECT / ".venv/pyvenv.cfg")),
        "cuda_library_candidates": [metadata(p) for p in LIBRARIES],
        "packaged_cuda_library_candidates": [
            metadata(str(path)) for folder in (
                PROJECT / ".venv/lib/python3.12/site-packages/nvidia",
                PROJECT / ".venv/lib/python3.12/site-packages/torch/lib")
            if folder.is_dir() for path in sorted(folder.rglob("*.so*"))
            if path.is_file()
        ],
        "site_pth_candidates": [
            {"path": str(p), "sha256": digest(p)}
            for p in sorted((PROJECT / ".venv/lib/python3.12/site-packages").glob("*.pth"))
            if p.is_file() and not p.is_symlink()
        ],
        "device_candidates_stat_only": [metadata(p) for p in DEVICES],
        "unresolved": [
            "static AST scan is not a complete import or dynamic-library census",
            "runtime and source directories recursively allowed by wrapper are not minimized or sealed",
            "GPU node visibility, CUDA initialization and device ioctl not tested",
            "network and IPC not blocked by the wrapper; external enforcement required",
            "actual host paths, immutable snapshots, denied-path probes and GPU coordination required",
            "prior outcome read contamination; exploratory only, no preregistration guarantee",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new candidate JSON; never overwrite")
    args = parser.parse_args()
    try:
        record = inventory()
        with open(args.output, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        parser.exit(1, f"inventory failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
