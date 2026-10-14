"""Content-addressed identities for research runs and stage inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fingerprint_file(path: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a stable file fingerprint; missing inputs are explicit."""

    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "size": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    }


def fingerprint_inputs(paths: Iterable[Path]) -> dict[str, Any]:
    records = sorted((fingerprint_file(Path(path)) for path in paths), key=lambda row: row["path"])
    return {"files": records, "sha256": sha256_bytes(canonical_json(records))}


def git_identity(repo: Path) -> dict[str, Any]:
    """Capture git provenance without making a non-git fixture unusable."""

    repo = Path(repo).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "HEAD", "--binary", "--"],
            check=True, capture_output=True,
        ).stdout
        return {"commit": commit, "dirty": dirty, "worktree_diff_sha256": sha256_bytes(diff)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": "unavailable"}


def build_run_identity(
    *,
    protocol_version: str,
    config: dict[str, Any],
    input_lock: dict[str, Any],
    code: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "protocol_version": protocol_version,
        "config": config,
        "input_lock": input_lock,
        "code": code,
    }
    return {"payload": payload, "sha256": sha256_bytes(canonical_json(payload))}
