"""Durable exact-prefix guard; the caller supplies authentic observations.

This module is not a physical runner or an authorization to execute one.
An identity is meaningful only when the caller has bound every field to the
actual environment, renderer and action history.
"""
import hashlib
import json
import os
from pathlib import Path
import re


_HEX = re.compile(r"^[0-9a-f]{64}$")


def identity(*, task_id, condition, prefix_hex, state, snapshot, pixels, history_hex):
    if not isinstance(task_id, str) or not _HEX.fullmatch(task_id):
        raise ValueError("invalid task identity")
    if condition not in ("off", "on"):
        raise ValueError("invalid condition")
    if not isinstance(prefix_hex, str) or not re.fullmatch(r"[0-9a-f]{80}", prefix_hex):
        raise ValueError("expected exact float32 prefix5 bytes")
    if not isinstance(history_hex, str) or not re.fullmatch(r"[0-9a-f]{240}", history_hex):
        raise ValueError("expected exact float32 warmup10 plus prefix5 bytes")
    if not history_hex.endswith(prefix_hex):
        raise ValueError("prefix and action history disagree")
    if not isinstance(pixels, dict) or set(pixels) != {"off", "on"} or any(
        not isinstance(v, str) or not _HEX.fullmatch(v) for v in pixels.values()
    ):
        raise ValueError("both physical pixel hashes required")
    if not isinstance(state, list) or len(state) != 7 or not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("native state and snapshot required")
    payload = dict(schema="p1_backtrack_cp5_identity_v1", task_id=task_id,
                   condition=condition, prefix_hex=prefix_hex, state=state,
                   snapshot=snapshot, pixels=pixels, history_hex=history_hex)
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("identity must be finite JSON") from exc


def establish_or_verify(path, payload):
    """Create an exclusive durable reference or check its exact bytes.

    A pre-existing mismatching, symlinked, or non-regular reference fails
    closed. Caller owns the directory and records the returned digest.
    """
    path = Path(path)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("identity bytes required")
    if path.is_symlink():
        raise ValueError("symlink reference rejected")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("prefix reference mismatch")
        return hashlib.sha256(payload).hexdigest()
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.is_symlink() or path.read_bytes() != payload:
        raise ValueError("prefix reference readback mismatch")
    return hashlib.sha256(payload).hexdigest()
