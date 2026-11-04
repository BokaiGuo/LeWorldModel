"""Live signal-driven paired P3 search using the existing native planner ABI.

Calling execute_host performs neural inference and requires separately
coordinated host CUDA. No result files, simulator steps, or training here.
Private callback seam is for CPU fixtures; it is not experimental evidence.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np

from . import codex_p3_ratio34_kernel_20261030_v1 as kernel
from . import p3_c3_dynamic_budget_20261102_exclusive as quota_kernel


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _execute_pair(point, tasks, banks, backends, meter, *, check_path, signal_fn, cost):
    """Consume signal, allocate, and perform real planner searches via callbacks."""
    _require(point in (20, 30) and type(point) is int, "fixed point")
    _require(len(tasks) == len(banks) == len(backends) == 2, "two roots required")
    _require(all(len(bank) >= point and [x["index"] for x in bank[:point]] == list(range(point))
                 for bank in banks), "shared ordered bank required")
    rows = []
    for arm in ("signal", "reverse"):
        weights = []
        for root in (0, 1):
            meter.scope = [point, arm, "signal", root]
            nominal = {"commands": [[0.0, 0.0] for _ in range(10)]}
            meter.emit("nominal_request", commands=nominal["commands"])
            with backends[root].guard(meter):
                path = backends[root].rollout(tasks[root], nominal["commands"], meter, "nominal")
                check_path(tasks[root], nominal["commands"], path)
                terminal, jacobian = meter.call(
                    "jacobian", lambda: backends[root].jacobian(
                        tasks[root], dict(nominal, contexts=path), meter))
            _require(np.array_equal(terminal, path[-1]["latent_history"][-1]),
                     "nominal terminal mismatch")
            result = meter.call("svd", lambda: signal_fn(tasks[root], terminal, jacobian))
            w = result.get("w") if isinstance(result, dict) else None
            _require(type(w) in (int, float) and math.isfinite(w) and w >= 0, "finite signal")
            weights.append(float(w))
            meter.emit("signal_complete", nominal_contexts=deepcopy(path),
                       terminal=terminal.tolist(), jacobian=kernel.native.blob(jacobian),
                       signal=result)
        _require(weights[0] != weights[1], "tied signal cannot produce distinct allocation")
        high = 0 if weights[0] > weights[1] else 1
        fixed = next(x for x in quota_kernel.protocol()["points"] if x["budget_point"] == point)
        sizes = sorted(fixed["signal"], reverse=True)
        allocation = [sizes[0], sizes[1]] if high == 0 else [sizes[1], sizes[0]]
        if arm == "reverse":
            allocation.reverse()
        meter.scope = [point, arm, "search", 0]
        allocation = meter.call("scheduler", lambda: allocation)
        meter.emit("allocation_frozen", quota=allocation, weights=weights)
        winners = []
        for root, count in enumerate(allocation):
            meter.scope = [point, arm, "search", root]
            candidates = []
            for index in range(count):
                item = banks[root][index]
                meter.emit("candidate_attempt", index=index, commands=item["commands"])
                with backends[root].guard(meter):
                    path = backends[root].rollout(tasks[root], item["commands"], meter, "search")
                check_path(tasks[root], item["commands"], path)
                value = meter.call("metric", lambda: cost(tasks[root], path))
                _require(type(value) in (int, float) and math.isfinite(value), "finite score")
                candidate = {"index": index, "commands": deepcopy(item["commands"]),
                             "cost": float(value), "path": deepcopy(path)}
                meter.emit("candidate_complete", candidate=candidate)
                candidates.append(candidate)
            winners.append(min(candidates, key=lambda x: (x["cost"], x["index"])))
        events = [event for event in meter.events if event["scope"][:2] == [point, arm]]
        attempted = Counter(event["resource"] for event in events)
        returned = Counter(event["resource"] for event in events if event["completed"])
        row = {"arm": arm, "weights": weights, "allocation": allocation,
               "winners": winners, "attempted": dict(attempted), "returned": dict(returned)}
        meter.emit("arm_complete", row=deepcopy(row))
        rows.append(row)
    a, b = rows
    _require(a["weights"] == b["weights"], "signal mismatch between arms")
    _require(a["allocation"] == b["allocation"][::-1], "allocations not opposed")
    _require(a["attempted"] == a["returned"] == b["attempted"] == b["returned"],
             "recorded resource axes not equal")
    return {"point": point, "rows": rows, "status": "paired_planning_completed",
            "equal_recorded_axes": True, "equal_total_compute": False,
            "physical_outcome": "not_measured", "source_authentication": "caller_responsibility",
            "P3_admission": False, "C3_admission": False}


def execute_host(model, device, tasks, sink, limits, sync, *, point):
    """Execute real JEPA planning only on host CUDA; caller manages safe output."""
    _require(getattr(device, "type", None) == "cuda", "host CUDA only")
    _require(all(callable(value) for value in (sink, limits, sync)), "required callbacks")
    meter = kernel.Meter(sink, limits, sync)
    backends = [kernel.native.Native(model, device, task) for task in tasks]
    banks = kernel.banks(kernel.native.BALL)
    try:
        return _execute_pair(point, tasks, banks, backends, meter,
                             check_path=kernel.native.check_path,
                             signal_fn=kernel.native.signal, cost=kernel.native.cost)
    except BaseException as exc:
        meter.failed = True
        meter.emit("experiment_failure", error_type=type(exc).__name__,
                   message=str(exc)[:1000], retry_authorized=False)
        raise
