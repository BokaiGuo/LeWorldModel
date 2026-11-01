"""Two-root fixed-panel bridge to the existing native H2 rollout.

This module is not a host launcher and does not authenticate its caller or
event sink. Importing it does not initialize CUDA or construct an environment.
"""
from collections import Counter
from copy import deepcopy
import math
import time

from experiments.spectral_reachability import codex_p3_fixed_panel_quota_20261030_v1 as quota
from experiments.spectral_reachability import agent_p3_resume_20261027_frontend as source
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as native

AXES = ("action_encoder", "predict", "predictor", "pred_proj")
PER_CANDIDATE = {axis: 2 for axis in AXES}
POLICIES = quota.POLICIES


class Meter:
    """Predispatch non-refundable calls, synchronized return; sink owns durability."""
    def __init__(self, sink, limits, sync):
        if not all(callable(x) for x in (sink, limits, sync)):
            raise ValueError("meter callbacks")
        self.sink, self.limits, self.sync = sink, limits, sync
        self.events, self.scope, self.forbidden_visual = [], None, []
        self.failed = False

    def emit(self, kind, **fields):
        self.sink(dict(kind=kind, scope=deepcopy(self.scope), **fields))

    def begin(self, resource):
        self.limits()
        if self.failed or resource not in AXES or self.scope is None:
            raise ValueError("meter resource or scope")
        if sum(e["resource"] == resource for e in self.events) >= 180:
            raise ValueError("global axis cap")
        if sum(e["resource"] == resource and e["scope"] == self.scope for e in self.events) >= 2:
            raise ValueError("candidate axis cap")
        entry = dict(index=len(self.events), resource=resource, scope=deepcopy(self.scope),
                     samples=1, completed=False, started_seconds=time.monotonic())
        self.events.append(entry)
        self.emit("resource_attempt", entry=deepcopy(entry))
        return entry

    def end(self, entry):
        if entry["completed"]:
            raise ValueError("duplicate return")
        self.sync()
        self.limits()
        entry["completed"] = True
        entry["ended_seconds"] = time.monotonic()
        self.emit("resource_return", entry=deepcopy(entry))

    def call(self, resource, fn):
        entry = self.begin(resource)
        try:
            out = fn()
            self.end(entry)
            return out
        except BaseException:
            self.failed = True
            raise


def _execute(tasks, banks, weights, backends, meter, *, check_path, cost):
    """Fixed three-arm order; each root has a separate winner."""
    if len(tasks) != 2 or len(banks) != 2 or len(backends) != 2:
        raise ValueError("two roots")
    if any(len(bank) < 15 for bank in banks):
        raise ValueError("15 candidates per root")
    for bank in banks:
        for i, candidate in enumerate(bank[:15]):
            if type(candidate.get("index")) is not int or candidate["index"] != i:
                raise ValueError("ordered candidate index")
            source.raw_commands(candidate["commands"])
    allocations = {p: quota.allocation(p, weights) for p in POLICIES}
    rows = []
    meter.emit("fixed_roster", policies=list(POLICIES), allocations=allocations,
               candidate_order=[[g, i] for g, i in quota.PANEL],
               task_hashes=[source.digest(t) for t in tasks],
               bank_hashes=[source.digest(b[:15]) for b in banks])
    try:
        for policy in POLICIES:
            group_rows = [[], []]
            for group, index in quota.PANEL:
                meter.scope = [policy, group, index]
                item = banks[group][index]
                before = len(meter.events)
                meter.emit("candidate_attempt", candidate_sha256=source.digest(item))
                with backends[group].guard(meter):
                    path = backends[group].rollout(tasks[group], item["commands"], meter, "search")
                check_path(tasks[group], item["commands"], path)
                value = cost(tasks[group], path)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("finite model cost")
                entries = meter.events[before:]
                attempted = Counter(e["resource"] for e in entries)
                completed = Counter(e["resource"] for e in entries if e["completed"])
                if dict(attempted) != PER_CANDIDATE or dict(completed) != PER_CANDIDATE:
                    raise ValueError("actual per-candidate operation vector")
                row = dict(policy=policy, group=group, index=index,
                           candidate_sha256=source.digest(item), commands=deepcopy(item["commands"]),
                           path=deepcopy(path), cost=value,
                           eligible=index < allocations[policy][group],
                           entry_start=before, entry_end=len(meter.events))
                group_rows[group].append(row)
                meter.emit("candidate_complete", row=deepcopy(row))
            winners = [min((r for r in group if r["eligible"]),
                           key=lambda r: (r["cost"], r["index"])) for group in group_rows]
            rows.append(dict(policy=policy, quotas=list(allocations[policy]),
                             candidates=group_rows, winners=deepcopy(winners)))
            meter.emit("arm_complete", policy=policy, winners=deepcopy(winners))
        meter.scope = None
        meter.emit("panel_complete", arm_count=len(rows), scientific_pass=False)
        return rows
    except BaseException as exc:
        meter.failed = True
        try:
            meter.emit("panel_failure", type=type(exc).__name__, message=str(exc)[:2048],
                       entry_count=len(meter.events), completed_arms=len(rows), retry_authorized=False)
        except BaseException as sink_error:
            exc.add_note("failure event not durable: " + repr(sink_error))
        raise


def execute_host(model, device, tasks, banks, weights, sink, limits, sync):
    """Real native H2 bridge. Host must provide source, lock and durable journal."""
    if getattr(device, "type", None) != "cuda":
        raise ValueError("host CUDA only")
    backends = [native.Native(model, device, task) for task in tasks]
    meter = Meter(sink, limits, sync)
    return _execute(tasks, banks, weights, backends, meter,
                    check_path=native.check_path, cost=native.cost)
