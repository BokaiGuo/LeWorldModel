"""Pre-execution source and checkpoint inventory; no model or CUDA loading.

An inventory of already imported Python files is not a full dynamic import
closure. The runtime must separately seal actual loaded modules and model.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys


def file_identity(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("canonical absolute file path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("regular file required")
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("file changed during hashing")
        return {"path": str(path), "sha256": h.hexdigest(), "bytes": before.st_size}
    finally:
        os.close(fd)


def inventory(paths):
    result = [file_identity(Path(p)) for p in paths]
    if not result or len({x["path"] for x in result}) != len(result):
        raise ValueError("nonempty unique inputs")
    return sorted(result, key=lambda row: row["path"])


def loaded_sources(prefixes):
    """Snapshot imported modules under explicitly specified canonical roots."""
    roots = [Path(p) for p in prefixes]
    if not roots or any(not p.is_absolute() or p.resolve() != p for p in roots):
        raise ValueError("canonical source roots")
    paths = set()
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == ".pyc" and path.with_suffix(".py").exists():
            path = path.with_suffix(".py")
        if path.is_file() and any(path.is_relative_to(root) for root in roots):
            paths.add(path)
    return inventory(sorted(paths))


def verify(expected):
    if type(expected) is not list or not expected:
        raise ValueError("source inventory")
    if inventory([r["path"] for r in expected]) != expected:
        raise ValueError("source or checkpoint drift")
