"""Atomic artifact writing and conservative output validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def artifact_is_valid(
    path: Path,
    *,
    required_keys: Iterable[str] = (),
    json_object: bool | None = None,
) -> tuple[bool, str]:
    """Validate existence, non-empty content, JSON shape and required keys."""

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing_or_empty"
    if path.suffix.lower() not in {".json", ".jsonl"}:
        return True, "non_json_nonempty"
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False, "invalid_json"
    if json_object is True and not isinstance(value, dict):
        return False, "json_not_object"
    if json_object is False and isinstance(value, dict):
        return False, "unexpected_json_object"
    missing = [key for key in required_keys if not isinstance(value, dict) or key not in value]
    if missing:
        return False, f"missing_keys:{','.join(missing)}"
    return True, "valid"
