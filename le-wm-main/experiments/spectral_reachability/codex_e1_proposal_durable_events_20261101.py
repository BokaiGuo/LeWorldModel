"""Durable, new-run proposal events; this does not reconstruct historical costs.

Create one journal per run in an existing directory. Each completed append is
fsynced before returning; the attempt context must surround the actual work.
An interrupted operation remains an unmatched attempt, not a zero cost.
Single writer only; no resume or repair of partial journals is attempted.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def read_events(path):
    """Validate the entire chain; truncated or modified journals are rejected."""
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete journal tail")
    previous = "0" * 64
    events = []
    for index, line in enumerate(raw.splitlines()):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        try:
            event = json.loads(line, object_pairs_hook=pairs,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(event, dict) or set(event) != {"index", "previous", "payload", "sha256"}:
                raise ValueError("invalid event fields")
            body = {k: event[k] for k in ("index", "previous", "payload")}
            digest = hashlib.sha256(_encode(body)).hexdigest()
            if type(event["index"]) is not int or event["index"] != index or event["previous"] != previous or event["sha256"] != digest:
                raise ValueError("invalid event chain")
            if _encode(event) != line:
                raise ValueError("noncanonical event")
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            raise ValueError("invalid journal event") from exc
        events.append(event)
        previous = digest
    return events


class ProposalJournal:
    """Exclusive new journal. A failed write permanently disables this writer."""

    def __init__(self, path, *, run_id, arm, proposal_kind):
        if not all(isinstance(x, str) and x for x in (run_id, arm, proposal_kind)):
            raise ValueError("run_id, arm and proposal_kind must be nonempty strings")
        self.path = Path(path)
        self.identity = {"run_id": run_id, "arm": arm, "proposal_kind": proposal_kind}
        self.index = 0
        self.tail = "0" * 64
        self.poisoned = False
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        self.fd = os.open(self.path, flags, 0o600)
        try:
            os.fsync(self.fd)
            directory_fd = os.open(self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def append(self, payload):
        if self.fd is None or self.poisoned:
            raise RuntimeError("journal closed or write failed")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        body = {"index": self.index, "previous": self.tail,
                "payload": {"identity": self.identity, "event": payload}}
        digest = hashlib.sha256(_encode(body)).hexdigest()
        line = _encode({**body, "sha256": digest}) + b"\n"
        try:
            offset = 0
            while offset < len(line):
                written = os.write(self.fd, line[offset:])
                if written <= 0:
                    raise OSError("short journal write")
                offset += written
            os.fsync(self.fd)
        except BaseException:
            self.poisoned = True
            raise
        self.index += 1
        self.tail = digest
        return body["index"]

    @contextmanager
    def attempt(self, operation, *, details=None):
        """Persist attempt before work, then success or exception without costs invented."""
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a nonempty string")
        attempt_index = self.append({"type": "attempt", "operation": operation, "details": details})
        try:
            yield attempt_index
        except BaseException as exc:
            try:
                self.append({"type": "failure", "attempt_index": attempt_index,
                             "exception_type": type(exc).__name__, "message": str(exc)})
            except BaseException as journal_exc:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"failure event could not be persisted: {journal_exc!r}")
            raise
        else:
            self.append({"type": "return", "attempt_index": attempt_index})

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
