"""Online signal-inclusive vector-cap experiment kernel; no production launcher.

Signal methods actually compute their own two root signals. Uniform gets no
signal and spends the eight saved forward calls on four extra candidates.
Calling execute_host requires a CUDA model and a durable external event sink.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
import time

import numpy as np
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as native

POINTS = (20, 30)
ARMS = ("uniform", "signal", "reverse")
NEURAL = ("action_encoder", "predict", "predictor", "pred_proj")
AD = ("jacobian", "vjp_directions", "grad_api", "raw_backward_hook", "svd")
AXES = NEURAL + AD + ("metric",)
TAG = "p3-online-budget-v1-cold-pcg64"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def caps(point):
    require(type(point) is int and point in POINTS, "fixed budget point")
    return {**{k: 2 * point + 8 for k in NEURAL}, "jacobian": 2,
            "vjp_directions": 384, "grad_api": 384, "raw_backward_hook": 384,
            "svd": 2, "metric": point + 4}


def allocation(point, arm, weights=None):
    caps(point)
    require(arm in ARMS, "arm")
    if arm == "uniform":
        require(weights is None, "uniform cannot access signal")
        n = point + 4
        return [n // 2, n // 2]
    require(type(weights) in (list, tuple) and len(weights) == 2 and
            all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in weights),
            "actual finite online weights")
    if weights[0] == weights[1]:
        return [point // 2, point // 2]
    high = 0 if weights[0] > weights[1] else 1
    if arm == "reverse":
        high = 1 - high
    n = point * 3 // 4
    return [n, point - n] if high == 0 else [point - n, n]


def banks(ball):
    require(math.isfinite(ball) and ball > 0, "ball")
    result = []
    for root in (0, 1):
        seed = int.from_bytes(hashlib.sha256(f"{TAG}:{root}".encode()).digest()[:8], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        rows = []
        for index in range(30):
            packets = []
            for _ in range(2):
                direction = rng.standard_normal(10)
                norm = np.linalg.norm(direction)
                require(math.isfinite(norm) and norm > 0, "invalid draw; no redraw")
                packets.append(direction / norm * ball * rng.random() ** (1 / 10))
            logits = np.asarray(packets, dtype=np.float32)
            raw = np.tanh(logits).reshape(10, 2)
            native.legal(raw, logits)
            rows.append(dict(index=index, commands=raw.astype(float).tolist(),
                             logits=logits.astype(float).tolist(), kind="online_budget_pcg64"))
        result.append(rows)
    return result


class Meter:
    """Each dispatch consumes its own non-refundable axis quota."""
    def __init__(self, sink, limits, sync):
        require(all(callable(x) for x in (sink, limits, sync)), "callbacks")
        self.sink, self.limits, self.sync = sink, limits, sync
        self.scope = None
        self.events = []
        self.failed = False
        self.forbidden_visual = []

    def emit(self, kind, **fields):
        self.sink(dict(kind=kind, scope=deepcopy(self.scope), **fields))

    def begin(self, resource):
        self.limits()
        require(not self.failed and resource in AXES and type(self.scope) is list and
                len(self.scope) == 4, "active scope and axis")
        point, arm, phase, root = self.scope
        require(arm in ARMS and phase in ("signal", "search") and
                type(root) is int and root in (0, 1), "arm/phase/root")
        require(not (arm == "uniform" and phase == "signal") and
                not (phase == "search" and resource in AD) and
                not (phase == "signal" and resource == "metric"), "resource phase")
        ceiling = caps(point)[resource]
        require(sum(e["resource"] == resource and e["scope"][:2] == [point, arm]
                    for e in self.events) < ceiling, "per-arm vector cap")
        entry = dict(index=len(self.events), resource=resource, scope=deepcopy(self.scope),
                     samples=1, completed=False, started_seconds=time.monotonic())
        self.events.append(entry)
        self.emit("resource_attempt", entry=deepcopy(entry))
        return entry

    def end(self, entry):
        require(not entry["completed"], "duplicate completion")
        self.sync()
        self.limits()
        entry["completed"] = True
        entry["ended_seconds"] = time.monotonic()
        self.emit("resource_return", entry=deepcopy(entry))

    def call(self, resource, fn):
        try:
            entry = self.begin(resource)
            value = fn()
            self.end(entry)
            return value
        except BaseException:
            self.failed = True
            raise


def execute(tasks, bank, backends, meter, *, check_path, signal_fn, cost):
    require(len(tasks) == len(bank) == len(backends) == 2 and
            all(len(b) == 30 and [x["index"] for x in b] == list(range(30)) for b in bank),
            "fixed two root bank")
    meter.emit("experiment_started", points=list(POINTS), arms=list(ARMS),
               caps={str(p): caps(p) for p in POINTS}, bank_tag=TAG,
               signal_sharing=False, scientific_pass=False)
    rows = []
    try:
        for point in POINTS:
            for arm in ARMS:
                weights = None
                if arm != "uniform":
                    weights = []
                    for root in (0, 1):
                        meter.scope = [point, arm, "signal", root]
                        first = len(meter.events)
                        zero = dict(commands=np.zeros((10, 2), dtype=np.float32).tolist())
                        meter.emit("nominal_request", commands=zero["commands"])
                        with backends[root].guard(meter):
                            path = backends[root].rollout(tasks[root], zero["commands"], meter, "nominal")
                            check_path(tasks[root], zero["commands"], path)
                            terminal, jacobian = meter.call(
                                "jacobian", lambda: backends[root].jacobian(
                                    tasks[root], dict(zero, contexts=path), meter))
                        require(np.array_equal(terminal, path[-1]["latent_history"][-1]), "nominal terminal")
                        result = meter.call("svd", lambda: signal_fn(tasks[root], terminal, jacobian))
                        vector = Counter(e["resource"] for e in meter.events[first:])
                        returned = Counter(e["resource"] for e in meter.events[first:] if e["completed"])
                        expected_signal = {**{k: 4 for k in NEURAL}, "jacobian": 1,
                                           "vjp_directions": 192, "grad_api": 192,
                                           "raw_backward_hook": 192, "svd": 1}
                        require(dict(vector) == dict(returned) == expected_signal, "actual root signal vector")
                        require(type(result) is dict and type(result.get("w")) in (float, int) and
                                math.isfinite(result["w"]) and result["w"] >= 0, "finite signal")
                        meter.emit("signal_complete", nominal_contexts=path, terminal=terminal.tolist(),
                                   jacobian=native.blob(jacobian), signal=result)
                        weights.append(result["w"])
                quota = allocation(point, arm, weights)
                meter.scope = [point, arm, "search", 0]
                meter.emit("allocation_frozen", quota=quota, weights=weights)
                winners = []
                for root in (0, 1):
                    meter.scope = [point, arm, "search", root]
                    candidates = []
                    for index in range(quota[root]):
                        item = bank[root][index]
                        first = len(meter.events)
                        meter.emit("candidate_attempt", index=index, commands=item["commands"])
                        with backends[root].guard(meter):
                            path = backends[root].rollout(tasks[root], item["commands"], meter, "search")
                        check_path(tasks[root], item["commands"], path)
                        value = meter.call("metric", lambda: cost(tasks[root], path))
                        vector = Counter(e["resource"] for e in meter.events[first:])
                        returned = Counter(e["resource"] for e in meter.events[first:] if e["completed"])
                        require(dict(vector) == dict(returned) == {**{k: 2 for k in NEURAL}, "metric": 1},
                                "actual candidate vector")
                        require(type(value) in (int, float) and math.isfinite(value), "finite cost")
                        candidate = dict(index=index, commands=deepcopy(item["commands"]),
                                         cost=value, path=path)
                        meter.emit("candidate_complete", candidate=candidate)
                        candidates.append(candidate)
                    winners.append(min(candidates, key=lambda x: (x["cost"], x["index"])))
                ledger = [e for e in meter.events if e["scope"][:2] == [point, arm]]
                actual = Counter(e["resource"] for e in ledger)
                returned = Counter(e["resource"] for e in ledger if e["completed"])
                expected = {k: 2 * point + 8 for k in NEURAL}
                expected.update({k: (0 if arm == "uniform" else caps(point)[k]) for k in AD})
                expected["metric"] = point + 4 if arm == "uniform" else point
                require(all(actual[k] == returned[k] == expected[k] for k in AXES), "actual arm census")
                row = dict(point=point, arm=arm, allocation=quota, weights=weights, winners=winners,
                           attempted={k: actual[k] for k in AXES},
                           returned={k: returned[k] for k in AXES})
                rows.append(row)
                meter.emit("arm_complete", row=deepcopy(row))
        meter.scope = None
        meter.emit("experiment_complete", arms=len(rows), scientific_pass=False)
        return rows
    except BaseException as exc:
        meter.failed = True
        try:
            meter.emit("experiment_failure", error_type=type(exc).__name__, message=str(exc)[:2048],
                       entries=len(meter.events), completed_arms=len(rows), retry_authorized=False)
        except BaseException as journal_error:
            exc.add_note("failure journal unavailable: " + repr(journal_error))
        raise


def execute_host(model, device, tasks, sink, limits, sync):
    require(getattr(device, "type", None) == "cuda", "host CUDA only")
    bank = banks(native.BALL)
    meter = Meter(sink, limits, sync)
    backends = [native.Native(model, device, task) for task in tasks]
    return execute(tasks, bank, backends, meter, check_path=native.check_path,
                   signal_fn=native.signal, cost=native.cost)
