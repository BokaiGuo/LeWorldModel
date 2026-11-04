"""Pre-outcome paired P3 allocation with durable classified call metering.

Callback seam only: no model import, device acquisition, physical execution,
or source/gate certification. The caller must use only the supplied adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

AXES = ("observation_encode", "action_encode", "predict_calls", "predict_samples",
        "signal_compute", "candidate_evaluate")
ARMS = ("signal", "reverse")
PHASES = ("signal", "search")
QUOTAS = {20: (15, 5), 30: (22, 8)}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class Journal:
    """Exclusive fsynced hash-chain; a sink failure poisons the session."""
    def __init__(self, path):
        path = Path(path)
        require(path.is_absolute() and path.parent.is_dir(), "absolute new journal required")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.file = os.fdopen(fd, "wb", buffering=0)
        self.previous = None
        self.index = 0
        self.failed = False
        try:
            dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except BaseException:
            self.file.close()
            raise

    def emit(self, kind, **values):
        require(not self.failed, "journal poisoned")
        require(not set(values) & {"index", "previous_sha256", "kind", "sha256"},
                "reserved journal fields")
        event = {"index": self.index, "previous_sha256": self.previous,
                 "kind": kind, **values}
        event["sha256"] = digest(event)
        line = canonical(event) + b"\n"
        try:
            if self.file.write(line) != len(line):
                raise OSError("short journal write")
            os.fsync(self.file.fileno())
        except BaseException:
            self.failed = True
            raise
        self.previous = event["sha256"]
        self.index += 1

    def close(self):
        self.file.close()


def freeze(point, caps, bank, binding):
    """Freeze contract and bank digest before callbacks or outcomes."""
    require(type(point) is int and point in QUOTAS, "only declared points 20 and 30")
    require(type(caps) is dict and set(caps) == set(AXES)
            and all(type(v) is int and v >= 0 for v in caps.values()), "complete nonnegative caps")
    require(type(binding) is dict and set(binding) == {"protocol", "model", "roots", "feedback"}
            and all(isinstance(binding[k], str) and binding[k] for k in binding), "binding required")
    require(type(bank) is list and len(bank) == 2, "two ordered banks required")
    for entries in bank:
        require(type(entries) is list and len(entries) >= QUOTAS[point][0], "bank too short")
        require(all(type(item) is dict and item.get("index") == i for i, item in enumerate(entries)),
                "candidate indices must be consecutive")
    return {"schema": "p3_preoutcome_classified_control.v1", "point": point,
            "caps_per_arm": dict(caps), "bank_sha256": digest(bank), "binding": dict(binding),
            "candidate_rule": "fixed indexed prefix; no outcome selection",
            "allocation_rule": "strict online signal rank; reverse opposite; reject ties",
            "phase_order": "all four signals, both allocations, then search",
            "equal_total_compute": False, "P3_admission": False, "C3_admission": False}


class MeteredModel:
    """Meter actual adapter-dispatched calls; raw-model bypass is not detected."""
    def __init__(self, model, caps, journal, arm, sync):
        self._model, self._sync, self._journal = model, sync, journal
        self.arm, self.caps = arm, dict(caps)
        self.attempted = dict.fromkeys(AXES, 0)
        self.returned = dict.fromkeys(AXES, 0)
        self.by_phase = {p: {"attempted": dict.fromkeys(AXES, 0),
                             "returned": dict.fromkeys(AXES, 0)} for p in PHASES}
        self.failed = False
        self._phase = None
        self._root = None

    def _dispatch(self, phase, charge, operation):
        require(not self.failed and phase in PHASES and phase == self._phase and self._root in (0, 1),
                "meter failed or invalid phase")
        require(set(charge) <= set(AXES) and all(type(x) is int and x > 0 for x in charge.values()),
                "invalid charge")
        if any(self.attempted[k] + v > self.caps[k] for k, v in charge.items()):
            self._journal.emit("cap_rejected", arm=self.arm, phase=phase, root=self._root, charge=charge,
                               attempted=dict(self.attempted))
            self.failed = True
            raise ValueError("resource cap exceeded before dispatch")
        self._journal.emit("call_intent", arm=self.arm, phase=phase, root=self._root, charge=charge)
        for k, v in charge.items():
            self.attempted[k] += v
            self.by_phase[phase]["attempted"][k] += v
        try:
            result = operation()
            self._sync()
        except BaseException as exc:
            self.failed = True
            self._journal.emit("call_failed", arm=self.arm, phase=phase, root=self._root, charge=charge,
                               error_type=type(exc).__name__, message=str(exc)[:500])
            raise
        self._journal.emit("call_returned", arm=self.arm, phase=phase, root=self._root, charge=charge)
        for k, v in charge.items():
            self.returned[k] += v
            self.by_phase[phase]["returned"][k] += v
        return result

    def encode(self, value, *, phase):
        return self._dispatch(phase, {"observation_encode": 1}, lambda: self._model.encode(value))

    def action_encoder(self, value, *, phase):
        return self._dispatch(phase, {"action_encode": 1}, lambda: self._model.action_encoder(value))

    def predict(self, context, actions, *, phase):
        shape = getattr(context, "shape", ())
        require(len(shape) > 0 and type(shape[0]) is int and shape[0] > 0, "predict batch required")
        return self._dispatch(phase, {"predict_calls": 1, "predict_samples": shape[0]},
                              lambda: self._model.predict(context, actions))

    def snapshot(self):
        return {"attempted": dict(self.attempted), "returned": dict(self.returned),
                "by_phase": self.by_phase, "caps": dict(self.caps), "failed": self.failed}


def run_pair(plan, bank, models, syncs, signal_fn, evaluate_fn, output):
    """Run two signal/reverse search arms; callbacks return only scalar signal/score.

    Signal callbacks must consume only pre-outcome root/feedback and the adapter.
    evaluate_fn sees one precommitted bank candidate, never future outcomes.
    Contract and callback source/closures require independent host certification.
    """
    require(plan == freeze(plan["point"], plan["caps_per_arm"], bank, plan["binding"]),
            "frozen plan or bank mismatch")
    plan = json.loads(canonical(plan))
    bank = json.loads(canonical(bank))
    require(type(models) is dict and set(models) == set(ARMS)
            and type(syncs) is dict and set(syncs) == set(ARMS)
            and all(callable(syncs[a]) for a in ARMS)
            and callable(signal_fn) and callable(evaluate_fn), "paired callbacks")
    j = Journal(output)
    meters = {a: MeteredModel(models[a], plan["caps_per_arm"], j, a, syncs[a]) for a in ARMS}
    try:
        j.emit("plan_frozen", plan=plan, plan_sha256=digest(plan))
        signals = {}
        for arm in ARMS:
            signals[arm] = []
            for root in (0, 1):
                m = meters[arm]
                m._phase = "signal"
                m._root = root
                before = m.returned["predict_calls"]
                result = m._dispatch("signal", {"signal_compute": 1},
                                     lambda: signal_fn(arm, root, m))
                require(m.returned["predict_calls"] > before, "signal must invoke metered predictor")
                require(type(result) in (int, float) and math.isfinite(result) and result >= 0,
                        "nonfinite/invalid signal")
                signals[arm].append(float(result))
                j.emit("signal_observed", arm=arm, root=root, value=float(result))
        require(signals["signal"] == signals["reverse"], "paired signal mismatch")
        require(signals["signal"][0] != signals["signal"][1], "tied signal")
        high = 0 if signals["signal"][0] > signals["signal"][1] else 1
        majority, minority = QUOTAS[plan["point"]]
        allocation = [majority, minority] if high == 0 else [minority, majority]
        allocations = {"signal": allocation, "reverse": list(reversed(allocation))}
        j.emit("both_allocations_frozen", signals=signals, allocations=allocations)
        winners = {}
        for arm in ARMS:
            winners[arm] = []
            for root in (0, 1):
                values = []
                for index in range(allocations[arm][root]):
                    item = json.loads(canonical(bank[root][index]))
                    j.emit("candidate_intent", arm=arm, root=root, candidate_index=index, candidate_sha256=digest(item))
                    m = meters[arm]
                    m._phase = "search"
                    m._root = root
                    before = m.returned["predict_calls"]
                    score = m._dispatch("search", {"candidate_evaluate": 1},
                                        lambda: evaluate_fn(arm, root, item, m))
                    require(m.returned["predict_calls"] > before, "candidate must invoke metered predictor")
                    require(type(score) in (int, float) and math.isfinite(score), "invalid candidate score")
                    j.emit("candidate_returned", arm=arm, root=root, candidate_index=index, score=float(score))
                    values.append((float(score), index))
                score, index = min(values)
                winners[arm].append({"root": root, "index": index, "score": score})
        result = {"status": "paired_callbacks_complete", "allocations": allocations, "winners": winners,
                  "resources": {a: meters[a].snapshot() for a in ARMS},
                  "equal_total_compute": False, "physical_outcome": "not_measured",
                  "raw_model_bypass": "not_detected", "P3_admission": False, "C3_admission": False}
        j.emit("pair_completed", result=result)
        return result
    except BaseException as exc:
        if not j.failed:
            j.emit("pair_failed", error_type=type(exc).__name__, message=str(exc)[:500],
                   resources={a: meters[a].snapshot() for a in ARMS}, retry_authorized=False)
        raise
    finally:
        j.close()
