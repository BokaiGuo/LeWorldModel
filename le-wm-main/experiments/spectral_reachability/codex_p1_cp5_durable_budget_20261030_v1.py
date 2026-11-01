"""Exclusive per-prefix axis ledger with pre-dispatch cap rejection."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .codex_p1_suffix_budget_outcome_20261030_v1 import AXES


class Budget:
    def __init__(self, path, caps, wall_seconds, *, clock=time.monotonic):
        path = Path(path)
        if (not path.is_absolute() or not path.parent.is_dir() or path.parent.resolve() != path.parent
                or type(caps) is not dict or set(caps) != set(AXES)
                or any(type(x) is not int or x < 0 for x in caps.values())
                or type(wall_seconds) not in (float, int)
                or not math.isfinite(wall_seconds) or wall_seconds <= 0):
            raise ValueError("invalid budget configuration")
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.path, self.caps, self.clock, self.start = path, dict(caps), clock, clock()
        self.wall_seconds, self.counts = wall_seconds, dict.fromkeys(AXES, 0)
        self.previous, self.seq, self.failed, self.closed, self.active = None, 0, False, False, False
        self._emit("header", {"caps": self.caps, "wall_seconds": wall_seconds})

    def _emit(self, kind, payload):
        body = {"seq": self.seq, "previous": self.previous, "kind": kind, "payload": payload}
        canonical = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(canonical(body)).hexdigest()
        raw = canonical({**body, "sha256": digest}) + b"\n"
        view = memoryview(raw)
        while view:
            n = os.write(self.fd, view)
            if n <= 0:
                raise OSError("short ledger write")
            view = view[n:]
        os.fsync(self.fd)
        self.previous, self.seq = digest, self.seq + 1

    def call(self, axis, units, fn):
        if self.closed or self.failed:
            raise ValueError("inactive budget")
        if self.active:
            self.failed = True
            self._emit("invalid", {"reason": "nested call"})
            raise ValueError("nested call")
        if axis not in AXES or type(units) is not int or units <= 0 or not callable(fn):
            self.failed = True
            self._emit("invalid", {})
            raise ValueError("invalid operation")
        event = {"axis": axis, "units": units}
        if self.clock() - self.start >= self.wall_seconds or self.counts[axis] + units > self.caps[axis]:
            self.failed = True
            self._emit("blocked", event)
            raise ValueError("wall or resource cap")
        self._emit("attempt", event)
        self.counts[axis] += units
        self.active = True
        try:
            value = fn()
            if self.failed:
                raise ValueError("nested call detected")
        except BaseException as exc:
            self.failed = True
            self._emit("failure", {**event, "error_type": type(exc).__name__, "message": str(exc)[:1000]})
            raise
        finally:
            self.active = False
        self._emit("return", event)
        return value

    def close(self):
        if self.closed or self.active:
            raise ValueError("closed budget")
        try:
            self._emit("terminal", {"status": "failed" if self.failed else "complete",
                                    "attempted": self.counts})
        finally:
            self.closed = True
            os.close(self.fd)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
