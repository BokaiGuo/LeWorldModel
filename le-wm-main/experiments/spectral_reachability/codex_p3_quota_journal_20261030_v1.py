"""Exclusive durable hash-chain for the fixed-panel experiment; no CUDA."""
import hashlib
import json
import os
from pathlib import Path
import stat

ZERO = "0" * 64


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


class Journal:
    def __init__(self, path, cap, reserve):
        if type(cap) is not int or type(reserve) is not int or not 0 < reserve < cap:
            raise ValueError("journal cap and terminal reserve")
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.cap, self.reserve, self.used = cap, reserve, 0
        self.index, self.tail, self.closed, self.broken = 0, ZERO, False, False
        try:
            d = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(d)
            finally:
                os.close(d)
        except BaseException:
            os.close(self.fd)
            self.closed = True
            raise

    def append(self, event, *, terminal=False):
        if self.closed or self.broken or type(event) is not dict:
            raise ValueError("journal closed, broken or invalid event")
        body = {"index": self.index, "previous": self.tail, "event": event}
        row = dict(body, sha256=digest(body))
        raw = encode(row) + b"\n"
        if self.used + len(raw) > self.cap - (0 if terminal else self.reserve):
            raise ValueError("journal cap")
        try:
            view = memoryview(raw)
            while view:
                count = os.write(self.fd, view)
                if count <= 0:
                    raise OSError("short journal write")
                view = view[count:]
            os.fsync(self.fd)
        except BaseException:
            self.broken = True
            raise
        self.used += len(raw)
        self.index += 1
        self.tail = row["sha256"]
        return self.tail

    def event(self, event):
        return self.append(event)

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def read(path, expected_tail, max_bytes):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            raise ValueError("journal file/cap")
        with os.fdopen(fd, "rb") as f:
            fd = -1
            raw = f.read(max_bytes + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > max_bytes or not raw.endswith(b"\n"):
        raise ValueError("incomplete journal")
    tail, events = ZERO, []
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    for i, line in enumerate(raw.splitlines()):
        row = json.loads(line, object_pairs_hook=pairs)
        if type(row) is not dict or set(row) != {"index", "previous", "event", "sha256"}:
            raise ValueError("journal row shape")
        body = {k: row[k] for k in ("index", "previous", "event")}
        if type(row["index"]) is not int or row["index"] != i or row["previous"] != tail or digest(body) != row["sha256"]:
            raise ValueError("journal hash chain")
        events.append(row["event"])
        tail = row["sha256"]
    if tail != expected_tail:
        raise ValueError("external journal tail")
    return events
