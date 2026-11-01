"""Exclusive bounded durable ledger for the preregistered dev-root producer.

This is infrastructure, not a physics producer or an authorization to run one.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


CAPS = {
    "reset": 12,
    "set_state": 24,
    "goal_setter": 16,
    "goal_bind": 4,
    "warmup_step": 60,
    "witness_step": 60,
    "physics_step": 1224,
    "explicit_render": 126,
    "goal_render": 12,
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class GuardError(RuntimeError):
    pass


class Ledger:
    """Durably register actual dispatches; leave an unmatched attempt on failure."""

    def __init__(self, directory, *, max_bytes=32 * 1024 * 1024,
                 reserve_bytes=512 * 1024, min_free_bytes=200 * 1024 * 1024,
                 wall_seconds=1200, clock=time.monotonic, disk_usage=shutil.disk_usage):
        self.directory = Path(directory)
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        self.min_free_bytes, self.wall_seconds = min_free_bytes, wall_seconds
        self.clock, self.disk_usage = clock, disk_usage
        self.started = clock()
        self.counts = {k: {"attempted": 0, "completed": 0} for k in CAPS}
        self.previous, self.index, self.bytes_written, self.failed = None, 0, 0, False
        if self.directory.exists():
            raise GuardError("output directory already exists")
        if disk_usage(self.directory.parent).free < min_free_bytes + max_bytes:
            raise GuardError("insufficient initial free disk")
        self.directory.mkdir()
        self._sync_dir()

    def _sync_dir(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _check(self, n, *, final=False):
        if self.failed:
            raise GuardError("ledger failed; preserve partial output and stop")
        if self.clock() - self.started >= self.wall_seconds:
            raise GuardError("wall budget")
        cap = self.max_bytes if final else self.max_bytes - self.reserve_bytes
        if self.bytes_written + n > cap:
            raise GuardError("output budget")
        if self.disk_usage(self.directory).free < self.min_free_bytes + n:
            raise GuardError("disk reserve")

    def write(self, name, data, *, final=False):
        if Path(name).name != name or name in ("", ".", "..", "journal.jsonl"):
            raise GuardError("unsafe or reserved filename")
        if not isinstance(data, bytes):
            raise TypeError("bytes required")
        self._check(len(data), final=final)
        try:
            with (self.directory / name).open("xb") as f:
                written = f.write(data)
                if written != len(data):
                    raise OSError("short write")
                f.flush()
                os.fsync(f.fileno())
            self.bytes_written += len(data)
            self._sync_dir()
        except OSError:
            self.failed = True
            raise
        return hashlib.sha256(data).hexdigest()

    def event(self, kind, **values):
        record = {"index": self.index, "previous_sha256": self.previous,
                  "kind": kind, **values}
        digest = hashlib.sha256(canonical(record)).hexdigest()
        line = canonical({**record, "sha256": digest}) + b"\n"
        self._check(len(line))
        try:
            with (self.directory / "journal.jsonl").open("ab") as f:
                written = f.write(line)
                if written != len(line):
                    raise OSError("short journal write")
                f.flush()
                os.fsync(f.fileno())
            self.bytes_written += len(line)
            self._sync_dir()
        except OSError:
            self.failed = True
            raise
        self.previous, self.index = digest, self.index + 1

    def call(self, resource, callback, **context):
        if resource not in CAPS:
            raise GuardError("unknown resource")
        count = self.counts[resource]
        if count["attempted"] >= CAPS[resource]:
            raise GuardError("resource cap: " + resource)
        self.event("attempt", resource=resource, **context)
        count["attempted"] += 1
        result = callback()
        count["completed"] += 1
        self.event("complete", resource=resource, **context)
        return result
