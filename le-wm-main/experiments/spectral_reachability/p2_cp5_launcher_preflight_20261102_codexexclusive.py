"""Fail-closed preparation for cp5 launcher; no physical execution entrypoint."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verified_source(physical, path, sha256, root_slot, task_id):
    """Reconstruct the original pinned prepare before constructing an env."""
    need(isinstance(sha256, str) and re.fullmatch("[0-9a-f]{64}", sha256),
         "prepare SHA")
    p, data = physical.verify(path, sha256)
    roots = data["roots"]
    need(p["source_slots"] == [r["slot"] for r in roots] and
         p["source_roots"] == physical.native.allmode.root_metadata(data) and
         p["warm_traces"] == [r["warm_trace"] for r in roots], "source reconstruction")
    matches = [x for x in p["plans"] if x is not None and x["root_slot"] == root_slot]
    need(matches and all(x["task_id"] == task_id for x in matches), "root/task binding")
    need(sum(r["slot"] == root_slot for r in roots) == 1, "root slot identity")
    return p, data, matches[0]


def validate_environment(physical, body):
    physical.validate_env(body)
    need(body.render_action is False and body.render_size == 224 and
         body.render_mode == "rgb_array", "native renderer identity")


class Ledger:
    """Exclusive fsynced append-only resource journal shared by sessions.

    Single-process only. Attempts are counted before callbacks, including failed
    callbacks. Limits must come from an independently frozen run contract.
    """
    def __init__(self, path, caps):
        self.path = Path(path)
        need(type(caps) is dict and caps and all(isinstance(k, str) and k and
             type(v) is int and v > 0 for k, v in caps.items()), "caps")
        self.caps = dict(caps)
        self.attempted = dict.fromkeys(caps, 0)
        self.returned = dict.fromkeys(caps, 0)
        self.previous = None
        self.sequence = 0
        self.poisoned = False
        self.file = self.path.open("xb")
        self.file.flush()
        os.fsync(self.file.fileno())
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._write("contract", caps=self.caps)

    def _write(self, kind, **fields):
        entry = {"index": self.sequence, "previous_sha256": self.previous, "kind": kind, **fields}
        raw = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
        data = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        try:
            need(self.file.write(data) == len(data), "short ledger write")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException:
            self.poisoned = True
            raise
        self.previous = entry["sha256"]
        self.sequence += 1

    def call(self, kind, fn, *, session):
        need(not self.poisoned and kind in self.caps and callable(fn) and
             isinstance(session, str) and session, "ledger route")
        if self.attempted[kind] >= self.caps[kind]:
            self._write("cap_rejected", resource=kind, session=session)
            raise ValueError("aggregate resource cap: " + kind)
        self._write("attempt", resource=kind, session=session)
        self.attempted[kind] += 1
        start = time.monotonic_ns()
        try:
            value = fn()
        except BaseException as exc:
            self._write("failed", resource=kind, session=session, elapsed_ns=time.monotonic_ns()-start,
                        error_type=type(exc).__name__, message=str(exc)[:500])
            raise
        self._write("returned", resource=kind, session=session, elapsed_ns=time.monotonic_ns()-start)
        self.returned[kind] += 1
        return value

    def close(self):
        self.file.close()


class AggregateMeter:
    """Wrap the historic Meter so nested native hooks hit aggregate caps."""
    def __init__(self, meter, ledger, session):
        self.meter, self.ledger, self.session = meter, ledger, session

    def __getattr__(self, name):
        return getattr(self.meter, name)

    @property
    def where(self):
        return self.meter.where

    @where.setter
    def where(self, value):
        self.meter.where = value

    def call(self, kind, fn, **kwargs):
        return self.ledger.call(kind, lambda: self.meter.call(kind, fn, **kwargs),
                                session=self.session)


def check_chain(path):
    """Check a closed ledger's byte-level hash chain, not its external identity."""
    previous = None
    count = 0
    with Path(path).open("rb") as source:
        for raw in source:
            need(raw.endswith(b"\n"), "ledger chain truncated")
            row = json.loads(raw)
            claimed = row.pop("sha256")
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            need(row["index"] == count and row["previous_sha256"] == previous and
                 claimed == hashlib.sha256(payload).hexdigest(), "ledger chain mismatch")
            previous, count = claimed, count + 1
    need(count > 0, "empty ledger chain")
    return {"count": count, "tail_sha256": previous}


def create_session_store(physical, path, ledger, session):
    """Use the historical exclusive/fsynced store and meter without altering it."""
    need(isinstance(ledger, Ledger) and isinstance(session, str) and session, "session ledger")
    store = physical.Store(path)
    meter = AggregateMeter(physical.Meter(store), ledger, session)
    store.emit("aggregate_binding", session=session, ledger_path=str(ledger.path),
               caps=deepcopy(ledger.caps), ledger_previous_sha256=ledger.previous)
    return store, meter


def require_feedback_winner(consumer, result_sha256, arm, env_factory):
    """No consumer for this specific feedback result exists: never authorize."""
    del consumer, result_sha256, arm, env_factory
    raise ValueError("feedback optimizer saved-output consumer not implemented; winner blocked")
