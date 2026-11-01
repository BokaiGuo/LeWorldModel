"""Development-only prospective physical callback accounting for legacy Meter."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import patch

from . import p1_prospective_total_guard_20261101_leibniz as total
from . import p1_moving_live_feedback_physical_20261027_v1 as legacy
from . import agent_p4_moving_multiboundary_20261028_v1 as moving

NAME = "p1_physical_callback_guard_20261101_leibniz"
RESOURCES = ("constructor", "reset", "set_state", "goal_setter", "goal_bind",
             "goal_render", "explicit_render", "public_step", "physics_step", "close")
# These are the only semantically corresponding axes in the unchanged total guard.
TOTAL_KIND = {"reset": "reset", "goal_render": "render", "explicit_render": "render",
              "public_step": "physical_dispatch", "physics_step": "physics_step"}
_LOCK = threading.Lock()


def _vector(value):
    total.require(type(value) is dict and set(value) == set(RESOURCES)
                  and all(type(v) is int and v >= 0 for v in value.values()), "physical vector")


class PhysicalGuard:
    """Second, source-bound vector and durable attempted/returned/failed journal."""
    def __init__(self, path, guard, caps, expected_contract_sha):
        _vector(caps)
        paths = (Path(__file__).resolve(), Path(legacy.__file__).resolve(), Path(moving.__file__).resolve())
        total.require(all(str(p) in guard.contract["source_sha256"] for p in paths), "physical source closure")
        total.check_sources(guard.contract)
        self.contract = {"schema": NAME + ".contract.v1", "total_contract_sha256": total.digest(guard.contract),
                         "caps": dict(caps)}
        total.require(total.digest(self.contract) == expected_contract_sha, "physical contract hash")
        path = Path(path)
        total.require(path.is_absolute() and path.parent.is_dir() and not path.exists() and not path.is_symlink(),
                      "exclusive physical journal")
        self.path, self.guard = path, guard
        self.caps, self.charged = dict(caps), dict.fromkeys(RESOURCES, 0)
        self.stack, self.failed, self.closed = [], False, False
        self.index, self.previous = 0, None
        self.stream = path.open("xb")
        self.emit("header", self.contract)

    def emit(self, kind, payload):
        event = {"index": self.index, "previous": self.previous, "kind": kind, "payload": payload}
        event["sha256"] = total.digest(event)
        data = total.canonical(event) + b"\n"
        total.require(self.stream.write(data) == len(data), "physical journal write")
        self.stream.flush()
        import os
        os.fsync(self.stream.fileno())
        self.index += 1
        self.previous = event["sha256"]
        return event["index"]

    def violation(self, message):
        self.failed = True
        self.emit("violation", {"reason": message})
        if not self.guard.failed and not self.guard.closed:
            self.guard.fail(message)
        raise ValueError(message)

    def dispatch(self, resource, callback):
        if self.closed or resource not in RESOURCES or not callable(callback):
            self.violation("physical operation/resource")
        # A cleanup close after an earlier failure is the sole fail-stop exception.
        if (self.failed or self.guard.failed) and resource != "close":
            self.violation("physical guard fail-stopped")
        if self.charged[resource] >= self.caps[resource]:
            self.failed = True
            self.emit("blocked", {"resource": resource, "reason": "physical cap"})
            if not self.guard.failed:
                self.guard.fail("physical cap exceeded before callback")
            raise ValueError("physical cap exceeded before callback")
        ident = self.emit("attempt", {"resource": resource, "parent": self.stack[-1] if self.stack else None})
        self.charged[resource] += 1
        self.stack.append(ident)
        try:
            if resource in TOTAL_KIND:
                value = self.guard.dispatch("physical", TOTAL_KIND[resource], 1, callback)
            else:
                value = callback()
            if resource != "close" and (self.failed or self.guard.failed):
                raise ValueError("physical guard violation during callback")
            self.emit("returned", {"id": ident})
            return value
        except BaseException as exc:
            self.failed = True
            self.emit("failure", {"id": ident, "type": type(exc).__name__, "message": str(exc)[:500]})
            if not self.guard.closed:
                self.guard.failed = True
            raise
        finally:
            self.stack.pop()

    def close(self):
        total.require(not self.closed and not self.stack, "physical close lifecycle")
        try:
            total.check_sources(self.guard.contract)
        except (ValueError, OSError):
            self.failed = True
            self.guard.failed = True
        self.emit("terminal", {"charged": self.charged, "status": "failed_or_blocked"
                               if self.failed or self.guard.failed else "closed_unverified",
                               "coverage_verified": False})
        self.stream.close()
        self.closed = True
        return {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "status": "failed_or_blocked" if self.failed or self.guard.failed else "closed_unverified"}


def verify_journal(path, expected_sha, expected_contract_sha):
    """Recompute journal chain, nested attempts, cap usage and terminal status."""
    data = Path(path).read_bytes()
    total.require(hashlib.sha256(data).hexdigest() == expected_sha and data.endswith(b"\n"), "physical journal hash")
    rows = [json.loads(line) for line in data.splitlines()]
    total.require(len(rows) >= 2 and rows[0]["kind"] == "header" and rows[-1]["kind"] == "terminal",
                  "physical journal lifecycle")
    contract = rows[0]["payload"]
    total.require(total.digest(contract) == expected_contract_sha
                  and contract["schema"] == NAME + ".contract.v1", "physical journal contract")
    _vector(contract["caps"])
    charged = dict.fromkeys(RESOURCES, 0)
    stack, previous, bad = [], None, False
    for i, row in enumerate(rows):
        copy = dict(row)
        sha = copy.pop("sha256")
        total.require(copy["index"] == i and copy["previous"] == previous and total.digest(copy) == sha,
                      "physical journal chain")
        previous = sha
        kind, payload = row["kind"], row["payload"]
        if kind == "attempt":
            resource = payload["resource"]
            total.require(resource in RESOURCES and payload["parent"] == (stack[-1] if stack else None)
                          and (not bad or resource == "close"), "physical attempt ordering")
            charged[resource] += 1
            total.require(charged[resource] <= contract["caps"][resource], "physical journal over cap")
            stack.append(i)
        elif kind in ("failure", "returned"):
            total.require(stack and payload["id"] == stack.pop(), "physical return ordering")
            if kind == "failure":
                bad = True
        elif kind in ("violation", "blocked"):
            bad = True
        else:
            total.require((i == 0 and kind == "header") or (i == len(rows)-1 and kind == "terminal"),
                          "physical unexpected event")
    terminal = rows[-1]["payload"]
    total.require(not stack and terminal["charged"] == charged and terminal["coverage_verified"] is False
                  and terminal["status"] in ("closed_unverified", "failed_or_blocked")
                  and (not bad or terminal["status"] == "failed_or_blocked"), "physical terminal")
    return {"charged": charged, "status": terminal["status"], "coverage_verified": False}


@contextmanager
def physical_meter_callbacks(physical, meter):
    """Scope actual Meter.call, including nested Meter.call from metered_class."""
    if not isinstance(meter, legacy.Meter):
        physical.violation("physical meter type")
    if not _LOCK.acquire(blocking=False):
        physical.violation("concurrent physical adapter")
    original_call, original_begin, original_end = meter.call, meter.begin, meter.end
    # Existing moving Meter lacks a close axis: add it only to this instance for
    # the scope, with the independently precommitted cap; never modify legacy.
    added_close = "close" not in meter.caps and "close" not in meter.counts
    if not added_close or any(key not in meter.caps or key not in meter.counts for key in RESOURCES if key != "close"):
        _LOCK.release()
        physical.violation("unexpected legacy physical axes")
    meter.caps["close"] = physical.caps["close"]
    meter.counts["close"] = {"attempted": 0, "completed": 0}
    charged_before = dict(physical.charged)
    meter_before = {key: meter.counts[key]["attempted"] for key in RESOURCES}
    frames = []

    def begin(resource, units=1):
        if not frames or frames[-1]["stage"] != "begin" or frames[-1]["resource"] != resource \
                or type(units) is not int or units != 1:
            physical.violation("direct physical meter begin bypass")
        ident = original_begin(resource, units)
        frames[-1].update(stage="callback", ident=ident)
        return ident

    def end(ident):
        if not frames or frames[-1]["stage"] != "end" or frames[-1]["ident"] != ident:
            physical.violation("direct physical meter end bypass")
        value = original_end(ident)
        frames[-1]["stage"] = "done"
        return value

    def call(resource, fn):
        if resource not in RESOURCES or not callable(fn):
            physical.violation("unknown physical callback")
        def invoke():
            frame = {"resource": resource, "stage": "begin", "ident": None}
            frames.append(frame)
            def wrapped():
                if frame["stage"] != "callback":
                    physical.violation("legacy meter begin missing")
                value = fn()
                frame["stage"] = "end"
                return value
            try:
                value = original_call(resource, wrapped)
                if frame["stage"] != "done":
                    physical.violation("legacy meter end missing")
                return value
            finally:
                frames.pop()
        return physical.dispatch(resource, invoke)

    try:
        with patch.object(meter, "call", call), patch.object(meter, "begin", begin), patch.object(meter, "end", end):
            yield meter
    finally:
        mismatch = not physical.failed and not physical.guard.failed and any(
            meter.counts[key]["attempted"] - meter_before[key] != physical.charged[key] - charged_before[key]
            for key in RESOURCES)
        del meter.caps["close"]
        del meter.counts["close"]
        _LOCK.release()
        if mismatch:
            physical.violation("physical meter accounting mismatch")
