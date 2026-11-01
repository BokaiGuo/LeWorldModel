"""Explicit H15 replay/physics/wall hook bridge; no GPU or environment launcher.

Start before standalone preparation, attach a fresh durable driver, and route
each replay and physical meter call through this bridge. Only passed callbacks
are observable. Time caps are checked at boundaries, not enforced in flight.
The physical hook uses the existing native_hooks meter ABI, without patching it.
"""
import copy
import hashlib
from pathlib import Path
import threading
import time

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec

NAME = "p2_h15_observable_budget_bridge_20261030_codex"
LIMIT = 16 * 1024 * 1024


def source_hash():
    return spec.sha(Path(__file__).read_bytes())


class Bridge:
    """Single-threaded sidecar with durable intent and fail-stop callbacks."""
    def __init__(self, directory, contract, contract_sha, arm, run_id, *, clock=time.monotonic_ns):
        spec.validate_contract(contract)
        core.require(spec.hash_string(contract_sha) and core.digest(contract) == contract_sha, "contract digest")
        core.require(type(arm) is str and arm in contract["arms"] and type(run_id) is str
                     and bool(run_id) and run_id.strip() == run_id, "arm/run identity")
        core.require(callable(clock), "monotonic clock")
        started = clock()
        core.require(type(started) is int and started >= 0, "clock origin")
        directory = Path(directory)
        core.require(directory.is_absolute() and not directory.exists() and not directory.is_symlink()
                     and directory.parent.is_dir(), "exclusive sidecar directory")
        directory.mkdir()
        self.path = directory
        self.clock, self.started, self.last = clock, started, started
        self.contract, self.contract_sha = copy.deepcopy(contract), contract_sha
        self.arm, self.run_id = arm, run_id
        self.journal = transport.Journal(directory / "budget_events.jsonl", LIMIT)
        self.lock = threading.RLock()
        self.driver = None
        self.failed = self.closed = self.planning_finished = False
        self.busy = False
        self.counts = {k: dict(attempted=0, completed=0) for k in ("replay_calls", "physical_steps")}
        self.journal.emit("header", dict(schema=NAME, contract=contract, contract_sha256=contract_sha,
                                         arm=arm, run_id=run_id, source_sha256=source_hash(),
                                         started_ns=started,
                                         wall_boundary="bridge creation through caller completion barrier",
                                         clock="caller-supplied monotonic_ns; production default time.monotonic_ns"))

    def _now(self):
        value = self.clock()
        core.require(type(value) is int and value >= self.last, "monotonic clock regression")
        self.last = value
        return value

    def attach(self, driver):
        try:
            core.require(type(driver) is durable.Driver and not driver.closed and not driver.failed
                         and self.driver is None and not self.failed and not self.closed
                         and not driver.ledger.receipt()["events"], "fresh driver before planning")
            rows = durable.read_chain((driver.directory / "events.jsonl").read_bytes())
            header = rows[0]["payload"]
            core.require(len(rows) == 1 and header["contract_sha256"] == self.contract_sha
                         and header["contract"] == self.contract and header["arm"] == self.arm
                         and header["run_id"] == self.run_id, "driver identity binding")
            self.journal.emit("attach", dict(driver_header_sha256=rows[0]["sha256"], at_ns=self._now()))
            self.driver = driver
        except BaseException:
            self.failed = True
            if type(driver) is durable.Driver:
                driver.failed = True
            raise

    def _call(self, axis, fn):
        core.require(self.lock.acquire(blocking=False), "concurrent bridge use forbidden")
        try:
            core.require(not self.closed and not self.failed and not self.busy and self.driver is not None,
                         "active nonrecursive bridge")
            core.require(callable(fn), "callback required")
            core.require((axis == "replay_calls" and not self.planning_finished and not self.driver.closed)
                         or (axis == "physical_steps" and self.planning_finished), "phase boundary")
            self.busy = True
            index = self.counts[axis]["attempted"]
            cap = self.contract["axes"][axis]["cap"]
            now = self._now()
            if index >= cap or (axis == "replay_calls" and now - self.started >= self.contract["axes"]["planning_wall_ns"]["cap"]):
                self.journal.emit("blocked", dict(axis=axis, index=index, at_ns=now, reason="cap_before_intent"))
                raise core.BudgetExceeded("bridge cap before " + axis)
            self.journal.emit("intent", dict(axis=axis, index=index, at_ns=now))
            dispatched = self._now()
            if axis == "replay_calls" and dispatched - self.started >= self.contract["axes"]["planning_wall_ns"]["cap"]:
                self.journal.emit("blocked", dict(axis=axis, index=index, at_ns=dispatched, reason="wall_after_intent"))
                raise core.BudgetExceeded("wall cap before replay callback")
            self.counts[axis]["attempted"] += 1
            try:
                result = fn()
                ended = self._now()
                if axis == "replay_calls" and ended - self.started > self.contract["axes"]["planning_wall_ns"]["cap"]:
                    raise core.BudgetExceeded("wall exceeded during replay")
                self.journal.emit("returned", dict(axis=axis, index=index, dispatched_ns=dispatched, at_ns=ended))
                self.counts[axis]["completed"] += 1
                return result
            except BaseException as exc:
                try:
                    self.journal.emit("failed", dict(axis=axis, index=index, dispatched_ns=dispatched,
                                                     at_ns=self._now(), error=type(exc).__name__))
                except BaseException:
                    pass
                raise
        except BaseException:
            self.failed = True
            if self.driver is not None and not self.driver.closed:
                self.driver.failed = True
            raise
        finally:
            self.busy = False
            self.lock.release()

    def replay(self, fn):
        """Wrap exactly one native replay in the existing durable replay axis."""
        index = self.counts["replay_calls"]["attempted"]
        scope = dict(stage="observable_replay", index=index, bridge_sha256=source_hash())
        return self._call("replay_calls", lambda: self.driver.call(scope, "replay", fn))

    def physical_meter(self, original):
        """Proxy for existing native_hooks; only physics_step uses the new cap."""
        return PhysicalMeter(self, original)

    def finish_planning(self, synchronize):
        """Caller-provided completion barrier is measured but cannot be attested."""
        core.require(self.lock.acquire(blocking=False), "concurrent bridge use forbidden")
        try:
            core.require(not self.closed and not self.failed and not self.busy and self.driver is not None
                         and not self.driver.closed and not self.planning_finished and callable(synchronize),
                         "active planning and explicit completion barrier")
            at = self._now()
            core.require(at - self.started < self.contract["axes"]["planning_wall_ns"]["cap"],
                         "planning wall before barrier")
            self.journal.emit("barrier_intent", dict(at_ns=at))
            core.require(self._now() - self.started < self.contract["axes"]["planning_wall_ns"]["cap"],
                         "planning wall before barrier callback")
            synchronize()
            end = self._now()
            core.require(end - self.started <= self.contract["axes"]["planning_wall_ns"]["cap"], "planning wall cap")
            self.journal.emit("planning_end", dict(at_ns=end, wall_ns=end-self.started,
                                                   barrier="caller_supplied_unverified"))
            self.planning_finished = True
            return end - self.started
        except BaseException:
            self.failed = True
            if self.driver is not None and not self.driver.closed:
                self.driver.failed = True
            raise
        finally:
            self.lock.release()

    def finish(self, driver_result):
        core.require(not self.closed and not self.busy and self.driver is not None and self.driver.closed,
                     "closed driver and inactive bridge required")
        try:
            expected_path = str(self.driver.directory / "events.jsonl")
            core.require(driver_result["journal"] == expected_path, "driver-local journal")
            check = durable.verify(expected_path, driver_result["sha256"], self.contract_sha)
            ok = not self.failed and self.planning_finished and check["status"] == "complete"
            self.journal.emit("terminal", dict(status="complete" if ok else "failed_or_blocked",
                                               ledger_sha256=driver_result["sha256"],
                                               counts=copy.deepcopy(self.counts), at_ns=self._now()))
        finally:
            self.closed = True
            self.journal.close()
        raw = (self.path / "budget_events.jsonl").read_bytes()
        return dict(journal=str(self.path / "budget_events.jsonl"), sha256=hashlib.sha256(raw).hexdigest(),
                    status="complete" if ok else "failed_or_blocked", coverage_verified=False,
                    budget_fairness=False, P1_admission=False, P2_admission=False, physical_release=False)


class PhysicalMeter:
    def __init__(self, bridge, original):
        self.bridge, self.original = bridge, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def call(self, kind, fn, **kwargs):
        try:
            if kwargs.get("cleanup") is not True or kind != "close":
                core.require(not self.bridge.failed and not self.bridge.closed and self.bridge.planning_finished,
                             "active physical phase")
            if kind == "physics_step":
                return self.original.call(kind, lambda: self.bridge._call("physical_steps", fn), **kwargs)
            return self.original.call(kind, fn, **kwargs)
        except BaseException:
            self.bridge.failed = True
            if self.bridge.driver is not None and not self.bridge.driver.closed:
                self.bridge.driver.failed = True
            raise


def verify(path, expected_sha, driver_path, driver_sha, contract_sha, *, arm, run_id):
    """Recount anchored sidecar and replay ledger, not unhooked work or time origin."""
    path = Path(path)
    core.require(spec.hash_string(expected_sha) and path.is_file() and not path.is_symlink()
                 and path.stat().st_size <= LIMIT, "bounded external sidecar digest")
    raw = path.read_bytes()
    core.require(len(raw) <= LIMIT and spec.sha(raw) == expected_sha, "sidecar digest")
    rows = durable.read_chain(raw)
    core.require(len(rows) >= 3 and rows[0]["kind"] == "header" and rows[1]["kind"] == "attach"
                 and rows[-1]["kind"] == "terminal", "sidecar lifecycle")
    h, a, terminal = rows[0]["payload"], rows[1]["payload"], rows[-1]["payload"]
    core.keys(h, ("schema", "contract", "contract_sha256", "arm", "run_id", "source_sha256",
                  "started_ns", "wall_boundary", "clock"))
    core.require(h["schema"] == NAME and h["source_sha256"] == source_hash()
                 and h["contract_sha256"] == contract_sha and core.digest(h["contract"]) == contract_sha
                 and h["arm"] == arm and h["run_id"] == run_id, "sidecar identity")
    spec.validate_contract(h["contract"])
    check = durable.verify(driver_path, driver_sha, contract_sha)
    driver_raw = Path(driver_path).read_bytes()
    core.require(spec.sha(driver_raw) == driver_sha, "ledger stable")
    driver_rows = durable.read_chain(driver_raw)
    core.keys(a, ("driver_header_sha256", "at_ns"))
    core.require(a["driver_header_sha256"] == driver_rows[0]["sha256"]
                 and driver_rows[0]["payload"]["arm"] == arm
                 and driver_rows[0]["payload"]["run_id"] == run_id, "driver header binding")
    core.keys(terminal, ("status", "ledger_sha256", "counts", "at_ns"))
    core.require(terminal["ledger_sha256"] == driver_sha and terminal["status"] in ("complete", "failed_or_blocked"),
                 "terminal ledger binding")
    counts = {k: dict(attempted=0, completed=0) for k in ("replay_calls", "physical_steps")}
    pending, end, blocked, barrier, replay_dispositions = None, None, False, False, []
    previous = h["started_ns"]
    core.require(type(previous) is int and previous >= 0, "clock origin")
    for row in rows[1:]:
        e, kind = row["payload"], row["kind"]
        core.require(type(e["at_ns"]) is int and e["at_ns"] >= previous, "monotonic event clock")
        previous = e["at_ns"]
        if kind == "attach":
            core.require(row is rows[1], "single attach")
        elif kind == "intent":
            core.keys(e, ("axis", "index", "at_ns"))
            axis = e["axis"]
            core.require(not blocked and pending is None and axis in counts
                         and (end is None) == (axis == "replay_calls")
                         and e["index"] == counts[axis]["attempted"]
                         and e["index"] < h["contract"]["axes"][axis]["cap"], "intent phase/cap")
            pending = e
        elif kind in ("returned", "failed"):
            core.keys(e, ("axis", "index", "dispatched_ns", "at_ns") if kind == "returned"
                      else ("axis", "index", "dispatched_ns", "at_ns", "error"))
            core.require(pending is not None and e["axis"] == pending["axis"] and e["index"] == pending["index"]
                         and type(e["dispatched_ns"]) is int
                         and pending["at_ns"] <= e["dispatched_ns"] <= e["at_ns"], "operation disposition")
            axis = e["axis"]
            if axis == "replay_calls":
                core.require(e["dispatched_ns"] - h["started_ns"] < h["contract"]["axes"]["planning_wall_ns"]["cap"],
                             "replay wall predispatch")
                if kind == "returned":
                    core.require(e["at_ns"] - h["started_ns"] <= h["contract"]["axes"]["planning_wall_ns"]["cap"],
                                 "replay wall return")
            counts[axis]["attempted"] += 1
            counts[axis]["completed"] += kind == "returned"
            if axis == "replay_calls":
                replay_dispositions.append(kind)
            pending = None
            blocked |= kind == "failed"
        elif kind == "blocked":
            core.keys(e, ("axis", "index", "at_ns", "reason"))
            axis = e["axis"]
            core.require(not blocked and axis in counts and e["index"] == counts[axis]["attempted"]
                         and (end is None) == (axis == "replay_calls")
                         and ((e["reason"] == "cap_before_intent" and pending is None and
                               (e["index"] >= h["contract"]["axes"][axis]["cap"] or
                                (axis == "replay_calls" and e["at_ns"]-h["started_ns"] >= h["contract"]["axes"]["planning_wall_ns"]["cap"])))
                              or (e["reason"] == "wall_after_intent" and pending is not None
                                  and pending["axis"] == axis == "replay_calls"
                                  and e["at_ns"]-h["started_ns"] >= h["contract"]["axes"]["planning_wall_ns"]["cap"])),
                         "actual cap block")
            pending, blocked = None, True
        elif kind == "barrier_intent":
            core.keys(e, ("at_ns",))
            core.require(not blocked and pending is None and end is None and not barrier, "planning barrier")
            barrier = True
        elif kind == "planning_end":
            core.keys(e, ("at_ns", "wall_ns", "barrier"))
            core.require(not blocked and pending is None and end is None and row["index"] > 2
                         and rows[row["index"]-1]["kind"] == "barrier_intent"
                         and e["barrier"] == "caller_supplied_unverified"
                         and e["wall_ns"] == e["at_ns"] - h["started_ns"]
                         and 0 <= e["wall_ns"] <= h["contract"]["axes"]["planning_wall_ns"]["cap"], "wall boundary")
            end = e["wall_ns"]
        else:
            core.require(kind == "terminal" and row is rows[-1], "unknown sidecar event")
    core.require(pending is None and terminal["counts"] == counts, "sidecar operation totals")
    ledger_ops = [e for e in driver_rows[-1]["payload"]["ledger"]["events"] if e["resource"] == "replay"]
    core.require(len(ledger_ops) == counts["replay_calls"]["attempted"]
                 and all(disposition != "returned" or op["status"] == "returned"
                         for disposition, op in zip(replay_dispositions, ledger_ops, strict=True))
                 and all(e["scope"] == dict(stage="observable_replay", index=i, bridge_sha256=source_hash())
                         for i, e in enumerate(ledger_ops)), "replay ledger bijection")
    if terminal["status"] == "complete":
        core.require(not blocked and end is not None and check["status"] == "complete", "complete sidecar")
    return dict(status=terminal["status"], counts=counts, planning_wall_ns=end,
                wall_boundary=h["wall_boundary"], barrier_verified=False, physical_hook_coverage_verified=False,
                replay_hook_coverage_verified=False, full_vector_coverage_verified=False,
                budget_fairness=False, P1_admission=False, P2_admission=False, physical_release=False)
