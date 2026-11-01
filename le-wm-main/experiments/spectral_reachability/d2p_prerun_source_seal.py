"""Bounded pre-inference source snapshots; no inference or weight discovery.

Call create_seal before inference, retain its manifest digest independently,
and call verify_seal after execution. A seal records bytes, not proof that a
runner executed those bytes. Failed creation leaves its exclusive directory
in place; never reuse it. Weight/config receipts may be supplied in args_json.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

MAX_FILES = 64
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_ARGS_BYTES = 256 * 1024
SCHEMA = "d2p_prerun_source_seal.v1"


def _path(value):
    if ".." in Path(value).parts:
        raise ValueError("parent traversal components forbidden")
    path = Path(os.path.abspath(os.fspath(value)))
    # Reject symlinks in ancestors as well as the final component.
    for part in (*reversed(path.parents), path):
        if part.is_symlink():
            raise ValueError(f"symlink paths forbidden: {part}")
    return path


def _read(path, limit):
    path = _path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"bounded regular file required: {path}")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"file exceeds size limit: {path}")
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def create_seal(*, files, args_json, output_dir):
    """Archive explicit small files and a JSON object into a NEW directory.

    args_json is a JSON string, not a pathname. Return the manifest SHA256 to
    retain outside this directory. No directory recursion or implicit inputs.
    """
    if not isinstance(args_json, str) or len(args_json.encode()) > MAX_ARGS_BYTES:
        raise ValueError("bounded args JSON string required")
    args = json.loads(args_json, object_pairs_hook=_unique_object)
    if not isinstance(args, dict):
        raise ValueError("args JSON must be an object")
    args_bytes = _json(args)  # Also rejects NaN/Infinity accepted by json.loads.
    if len(args_bytes) > MAX_ARGS_BYTES:
        raise ValueError("canonical args JSON exceeds limit")
    if isinstance(files, (str, bytes, Path)):
        raise ValueError("explicit file list required")
    paths = []
    for value in files:
        if len(paths) >= MAX_FILES:
            raise ValueError("too many source files")
        paths.append(_path(value))
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("nonempty unique explicit files required")
    blobs, total = [], 0
    for path in paths:
        data = _read(path, MAX_FILE_BYTES)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("total source size exceeds limit")
        blobs.append(data)
    output = _path(output_dir)
    output.mkdir(exist_ok=False)  # Parent must exist; never reuse an old seal.
    archive = output / "sources"
    archive.mkdir()
    entries = []
    for index, (path, data) in enumerate(zip(paths, blobs, strict=True)):
        name = f"sources/{index:03d}.source"
        with (output / name).open("xb") as stream:
            stream.write(data)
        entries.append({"source": str(path), "archive": name,
                        "size": len(data), "sha256": _sha(data)})
    with (output / "args.json").open("xb") as stream:
        stream.write(args_bytes)
    manifest = {"schema": SCHEMA, "files": entries,
                "args_sha256": _sha(args_bytes), "inference_executed": False}
    raw = _json(manifest)
    if len(raw) > MAX_ARGS_BYTES:
        raise ValueError("manifest exceeds size limit; seal incomplete")
    with (output / "manifest.json").open("xb") as stream:
        stream.write(raw)
    digest = _sha(raw)
    verify_seal(output_dir=output, expected_manifest_sha256=digest)
    return {"output_dir": str(output), "manifest_sha256": digest,
            "source_count": len(entries), "source_bytes": total}


def verify_seal(*, output_dir, expected_manifest_sha256):
    """Fail closed on manifest, args, archive, or current-source byte changes.

    This is an end-of-run content check, not continuous filesystem monitoring
    or protection against adversarial concurrent ancestor replacement.
    """
    output = _path(output_dir)
    raw = _read(output / "manifest.json", MAX_ARGS_BYTES)
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError("manifest SHA256 mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported source seal schema")
    entries = manifest["files"]
    if not 1 <= len(entries) <= MAX_FILES:
        raise ValueError("invalid source count")
    if _sha(_read(output / "args.json", MAX_ARGS_BYTES)) != manifest["args_sha256"]:
        raise ValueError("args SHA256 mismatch")
    total = 0
    for index, entry in enumerate(entries):
        if entry["archive"] != f"sources/{index:03d}.source":
            raise ValueError("invalid archive path")
        source = Path(entry["source"])
        if not source.is_absolute():
            raise ValueError("absolute source path required")
        for kind, path in (("source", source), ("archive", output / entry["archive"])):
            data = _read(path, MAX_FILE_BYTES)
            if len(data) != entry["size"] or _sha(data) != entry["sha256"]:
                raise ValueError(f"{kind} SHA256/size mismatch: {path}")
        total += entry["size"]
    if total > MAX_TOTAL_BYTES:
        raise ValueError("total source size exceeds limit")
    return {"status": "verified", "manifest_sha256": expected_manifest_sha256,
            "source_count": len(entries), "source_bytes": total,
            "scope": "source/archive/args bytes unchanged; not execution or mechanism proof"}
