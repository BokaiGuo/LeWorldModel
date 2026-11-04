"""Read-only old-source audit and new Python inventory; never authorizes execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[2]
OLD_PREPARE = ROOT / "results/codex_p3_online_budget_host_20261030_v1_RUN_01/prepare.json"
OLD_SHA256 = "bfab85393221afe02f8ac804130040c4f493d5d3deffdeb5486e14cc6e5cb305"
PACKAGES = (
    ("torch", "torch"), ("torchvision", "torchvision"),
    ("transformers", "transformers"), ("stable-pretraining", "stable_pretraining"),
    ("stable-worldmodel", "stable_worldmodel"), ("numpy", "numpy"),
)
NAME = "p3_environment_migration_preflight_20261102_codexexclusive"


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_sources(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty historical source inventory required")
    result, paths = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source row must be an object")
        path, expected, size = row.get("path"), row.get("sha256"), row.get("bytes")
        if (not isinstance(path, str) or not Path(path).is_absolute() or path in paths
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)
                or type(size) is not int or size < 0):
            raise ValueError("invalid historical source row")
        paths.add(path)
        p = Path(path)
        record = {"path": path, "expected_sha256": expected, "expected_bytes": size,
                  "actual_sha256": None, "actual_bytes": None}
        try:
            if p.is_symlink():
                record["status"] = "symlink_unverified"
            elif not p.exists():
                record["status"] = "missing"
            elif not p.is_file():
                record["status"] = "not_regular"
            else:
                record["actual_bytes"] = p.stat().st_size
                record["actual_sha256"] = sha_file(p)
                record["status"] = ("match" if record["actual_bytes"] == size
                                    and record["actual_sha256"] == expected else "mismatch")
        except OSError as exc:
            record["status"] = "unreadable"
            record["error_type"] = type(exc).__name__
        result.append(record)
    return result


def current_environment():
    packages = []
    for distribution, module in PACKAGES:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            spec = importlib.util.find_spec(module)
            location = None if spec is None else spec.origin
        except (ImportError, ValueError, OSError):
            location = None
        packages.append({"distribution": distribution, "module": module,
                         "version": version, "module_origin": location})
    executable = Path(sys.executable).resolve()
    return {"python_version": platform.python_version(), "executable": str(executable),
            "executable_sha256": sha_file(executable), "packages": packages,
            "dependency_closure_complete": False, "cuda_probed": False}


def legacy_closure_probe():
    """Inspect the earlier closure's explicit inputs; do not run closure or model."""
    try:
        from . import moving_history_budget_scheduler_cuda_20261027_v1 as legacy
        paths = set(legacy.blind.source_files() + legacy.moving.moving.source_files()
                    + [Path(legacy.__file__), Path(legacy.moving.__file__),
                       Path(legacy.authority.__file__), Path(legacy.fidelity.__file__),
                       legacy.PREREG, legacy.ROOT / "tests" / ("test_" + legacy.NAME + ".py"),
                       legacy.ROOT / "innerdocs" / (legacy.NAME + ".md")])
        missing = sorted(str(p) for p in paths if not Path(p).is_file() or Path(p).is_symlink())
        return {"status": "explicit_inputs_probed_only", "count": len(paths),
                "missing_or_symlink": missing, "transitive_closure_complete": False}
    except Exception as exc:
        return {"status": "probe_failed", "error_type": type(exc).__name__,
                "message": str(exc)[:1000], "transitive_closure_complete": False}


def build(old_path=OLD_PREPARE, expected_sha=OLD_SHA256):
    old_path = Path(old_path)
    if old_path.is_symlink() or not old_path.is_file():
        raise ValueError("historical prepare must be a regular non-symlink file")
    actual = sha_file(old_path)
    if actual != expected_sha:
        raise ValueError("historical prepare SHA mismatch")
    with old_path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema") != "codex_p3_online_budget_host_20261030_v1.prepare":
        raise ValueError("unexpected historical prepare schema")
    rows = audit_sources(plan.get("sources"))
    counts = {status: sum(row["status"] == status for row in rows) for status in (
        "match", "missing", "mismatch", "symlink_unverified", "not_regular", "unreadable")}
    return {
        "schema": NAME + ".v1", "historical_prepare": str(old_path),
        "historical_prepare_sha256": actual, "historical_sources": rows,
        "historical_source_counts": counts,
        "historical_source_list_complete": False,
        "historical_dynamic_source_closure": plan.get("dynamic_source_closure", "unknown"),
        "historical_model_expected_hash": plan.get("expected_model_hash"),
        "historical_task_sha256": plan.get("task_sha256"),
        "historical_bank_sha256": plan.get("bank_sha256"),
        "current_environment": current_environment(),
        "legacy_closure_probe": legacy_closure_probe(),
        "migration_status": "not_verified", "old_new_equivalence": "unknown",
        "task_bank_reconstructed": False, "model_tensor_hash_verified": False,
        "runtime_parity_verified": False, "execution_authorized": False,
        "cuda_used": False, "P3_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if (not path.is_absolute() or path.parent != ROOT / "results"
            or not path.name.startswith(NAME + "_") or os.path.lexists(path)):
        raise ValueError("new exclusive result path under results required")
    report = build()
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "historical_source_counts": report["historical_source_counts"],
                      "execution_authorized": False}))


if __name__ == "__main__":
    main()
