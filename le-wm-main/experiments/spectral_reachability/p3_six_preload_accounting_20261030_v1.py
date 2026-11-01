"""Offline-testable outer accounting seam; NOT a production launcher.

Callbacks are supplied by the caller. Nothing here imports a loader, torch or
physics, or establishes authority. Counters are observations, not attestation.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

NAME = "p3_six_preload_accounting_20261030_v1"
STAGES = ("cuda_context", "checkpoint_and_model_load", "guarded_six_host")
METRICS = ("rss_peak_bytes", "cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes",
           "logical_checkpoint_bytes")
UNKNOWN = {
    "pre_entry_process_cost": "unknown",
    "kernel_physical_io_bytes": "unknown",
    "driver_and_external_gpu_memory": "unknown",
    "continuous_resource_peaks": "unknown",
    "loader_read_coverage": "unknown",
    "gpu_kernels_and_flops": "unknown",
    "actual_hardware_identity": "unknown",
    "resource_fairness": "unknown",
    "host_plan_actually_enforced": "unknown",
    "journal_tail_wall_cost": "unknown",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def plan_check(plan, expected):
    need(type(plan) is dict and set(plan) == {"caps", "host_plan_sha256", "host_wall_ns",
         "source_sha256", "journal_cap_bytes", "failure_reserve_bytes"}, "explicit accounting plan")
    need(type(plan["caps"]) is dict and set(plan["caps"]) == {"wall_ns", *METRICS} and
         all(type(x) is int and x > 0 for x in plan["caps"].values()), "explicit positive ceilings")
    for key in ("host_plan_sha256", "source_sha256"):
        value = plan[key]
        need(type(value) is str and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), "SHA pin")
    need(type(plan["host_wall_ns"]) is int and 0 < plan["host_wall_ns"] <= plan["caps"]["wall_ns"],
         "host wall budget")
    need(type(plan["journal_cap_bytes"]) is int and type(plan["failure_reserve_bytes"]) is int and
         0 < plan["failure_reserve_bytes"] < plan["journal_cap_bytes"], "journal reservation")
    need(type(expected) is str and digest(plan) == expected, "accounting plan pin")


class Journal:
    """Exclusive append-only chained journal; fsync each event, preserving failures."""
    def __init__(self, path, cap, reserve):
        self.path = Path(path)
        need(self.path.is_absolute() and self.path.parent.is_dir() and
             self.path.parent.resolve() == self.path.parent and not self.path.exists(), "exclusive journal path")
        self.cap, self.reserve, self.used, self.index, self.previous = cap, reserve, 0, 0, None
        self.poisoned = False
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._syncdir()

    def _syncdir(self):
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def emit(self, kind, detail, *, failure=False):
        need(not self.poisoned, "poisoned journal")
        record = dict(index=self.index, previous=self.previous, kind=kind, detail=detail)
        record["sha256"] = digest(record)
        line = raw(record) + b"\n"
        need(len(line) <= 65536 and self.used + len(line) <= self.cap - (0 if failure else self.reserve),
             "journal ceiling")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            with os.fdopen(fd, "wb") as stream:
                need(stream.write(line) == len(line), "complete journal write")
                stream.flush()
                os.fsync(stream.fileno())
            self._syncdir()
        except BaseException:
            self.poisoned = True
            raise
        self.used += len(line)
        self.previous = record["sha256"]
        self.index += 1


def run_offline_seam(path, plan, plan_sha256, *, host_plan, clock_ns, sample, init_context, load_model, run_host):
    """Run injected callbacks under sampled limits. Never production authority.

    sample(stage, boundary) returns cumulative integer peaks/counters or None.
    GPU counters may be None only before context initialization. The host
    callback receives (model, host_plan, guard); current guarded host lacks this hook, so
    production integration requires separate work.
    """
    start = clock_ns()
    need(type(start) is int and start >= 0, "monotonic clock start")
    plan_check(plan, plan_sha256)
    plan = deepcopy(plan)
    need(type(host_plan) is dict and digest(host_plan) == plan["host_plan_sha256"] and
         type(host_plan.get("wall")) is int and host_plan["wall"] * 1_000_000_000 == plan["host_wall_ns"] and
         all(type(host_plan.get(k)) is int and host_plan[k] > 0 for k in
             ("cuda_allocated_bytes", "cuda_reserved_bytes")) and
         host_plan["cuda_allocated_bytes"] <= plan["caps"]["cuda_allocated_peak_bytes"] and
         host_plan["cuda_reserved_bytes"] <= plan["caps"]["cuda_reserved_peak_bytes"],
         "fixed host plan and whole-chain ceilings")
    host_plan = deepcopy(host_plan)
    journal = Journal(path, plan["journal_cap_bytes"], plan["failure_reserve_bytes"])
    previous = {key: None for key in METRICS}
    last_time = start
    stage = "setup"
    failed = False

    def guard(phase, boundary):
        nonlocal last_time, failed
        try:
            need(not failed, "accounting guard previously failed")
            now = clock_ns()
            need(type(now) is int and now >= last_time, "monotonic clock")
            last_time = now
            observed = sample(phase, boundary)
            need(type(observed) is dict and set(observed) == set(METRICS), "sample schema")
            elapsed = now - start
            detail = dict(phase=phase, boundary=boundary, elapsed_ns=elapsed, counters=observed)
            journal.emit("sample", detail)
            need(elapsed <= plan["caps"]["wall_ns"], "whole-chain wall exceeded")
            for key, value in observed.items():
                if value is None:
                    need(phase == "cuda_context" and boundary == "before" and key.startswith("cuda_"),
                         "required counter unknown: " + key)
                    continue
                need(type(value) is int and value >= 0, "invalid counter: " + key)
                need(previous[key] is None or value >= previous[key], "counter regression: " + key)
                need(value <= plan["caps"][key], "resource ceiling: " + key)
                previous[key] = value
            return elapsed
        except BaseException:
            failed = True
            raise

    try:
        journal.emit("begin", dict(plan=plan, plan_sha256=plan_sha256, unknown=UNKNOWN,
                                   authority=False, scientific_pass=False))
        model = None
        for stage, callback in zip(STAGES, (init_context, load_model, run_host), strict=True):
            elapsed = guard(stage, "before")
            if stage == "guarded_six_host":
                need(plan["caps"]["wall_ns"] - elapsed >= plan["host_wall_ns"],
                     "insufficient remaining whole-chain wall for fixed host plan")
            journal.emit("intent", dict(stage=stage))
            if stage == "cuda_context":
                callback(lambda: guard(stage, "inside"))
            elif stage == "checkpoint_and_model_load":
                model = callback(lambda: guard(stage, "inside"))
                need(model is not None, "missing loaded model")
            else:
                outcome = callback(model, deepcopy(host_plan), lambda: guard(stage, "inside"))
                need(type(outcome) is dict and set(outcome) == {"begin_sha256", "end_sha256",
                     "report_sha256", "completion_sha256"} and all(type(x) is str and len(x) == 64 and
                     all(c in "0123456789abcdef" for c in x) for x in outcome.values()),
                     "guarded host output pins")
            guard(stage, "after")
            journal.emit("return", dict(stage=stage))
        journal.emit("end", dict(host_output=outcome, unknown=UNKNOWN, accounting_complete=False,
                                  physical_certified=False, scientific_pass=False))
        return dict(journal_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    host_output=outcome, unknown=dict(UNKNOWN), accounting_complete=False,
                    physical_certified=False, scientific_pass=False)
    except BaseException as exc:
        try:
            journal.emit("failure", dict(stage=stage, type=type(exc).__name__, message=str(exc)[:2048]),
                         failure=True)
        except BaseException as secondary:
            exc.add_note("accounting failure journal unavailable: " + repr(secondary))
        raise
