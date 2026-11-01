"""Explicit P3 model-call adapter; no model, GPU or environment is created.

This is a prospective integration seam, not instrumentation of a raw model
reference. The executor must call this adapter for every in-scope operation.
"""
from __future__ import annotations

from copy import deepcopy

from . import p3_c3_dynamic_budget_20261102_exclusive as budget


AXES = ("action_encoder", "predict", "predictor_sample")
PHASES = ("signal", "search")


def _require(value, message):
    if not value:
        raise ValueError(message)


class BoundaryAdapter:
    """Meter the JEPA action_encoder and predict API boundaries.

    ``sync`` must wait for asynchronous device work; a sync failure leaves the
    corresponding attempt unreturned. The batch dimension of the latent
    context defines predictor samples, not the number of action tokens.
    """

    def __init__(self, model, *, point, arm, caps, sync):
        rows = {r["budget_point"]: r for r in budget.protocol()["points"]}
        _require(type(point) is int and point in rows and arm in ("signal", "reverse"),
                 "fixed point and arm")
        _require(type(caps) is dict and set(caps) == set(AXES)
                 and all(type(v) is int and v >= 0 for v in caps.values()),
                 "complete caps")
        _require(callable(sync) and callable(getattr(model, "action_encoder", None))
                 and callable(getattr(model, "predict", None)), "model ABI and sync")
        self._model = model
        self._sync = sync
        self.point = point
        self.arm = arm
        self.quotas = tuple(rows[point][arm])
        self.caps = dict(caps)
        self.attempted = dict.fromkeys(AXES, 0)
        self.returned = dict.fromkeys(AXES, 0)
        self.by_phase = {p: {"attempted": dict.fromkeys(AXES, 0),
                             "returned": dict.fromkeys(AXES, 0)} for p in PHASES}
        self.started = [0, 0]
        self.completed = [0, 0]
        self.events = []
        self.failed = False
        self._active = False

    def _call(self, phase, operation, charge):
        _require(not self.failed and phase in PHASES, "adapter closed or invalid phase")
        _require(all(self.attempted[k] + charge.get(k, 0) <= self.caps[k] for k in AXES),
                 "resource cap exceeded before dispatch")
        for key, amount in charge.items():
            self.attempted[key] += amount
            self.by_phase[phase]["attempted"][key] += amount
        event = {"phase": phase, "charge": dict(charge), "returned": False}
        self.events.append(event)
        try:
            result = operation()
            self._sync()
        except BaseException:
            self.failed = True
            raise
        for key, amount in charge.items():
            self.returned[key] += amount
            self.by_phase[phase]["returned"][key] += amount
        event["returned"] = True
        return result

    def predict(self, context, action_context, *, phase):
        """Call native encoder, then native predict, charging at each boundary."""
        _require(self._active, "candidate scope required")
        shape = getattr(context, "shape", ())
        _require(len(shape) >= 1 and type(shape[0]) is int and shape[0] > 0,
                 "positive context batch dimension")
        encoded = self._call(phase, lambda: self._model.action_encoder(action_context),
                             {"action_encoder": 1})
        return self._call(phase, lambda: self._model.predict(context, encoded),
                          {"predict": 1, "predictor_sample": shape[0]})

    def candidate(self, root, index, evaluate):
        """Evaluate each root's fixed prefix of candidate indices exactly once."""
        _require(not self.failed and not self._active, "adapter closed or nested scope")
        _require(type(root) is int and root in (0, 1)
                 and type(index) is int and index == self.started[root]
                 and index < self.quotas[root], "candidate quota/order")
        _require(callable(evaluate), "candidate callable")
        self.started[root] += 1
        self._active = True
        try:
            result = evaluate(self)
        except BaseException:
            self.failed = True
            raise
        finally:
            self._active = False
        self.completed[root] += 1
        return result

    def snapshot(self):
        return {
            "schema": "p3_c3_model_boundary_20261102_exclusive.v1",
            "point": self.point, "arm": self.arm, "allocation": list(self.quotas),
            "candidate_started": list(self.started), "candidate_returned": list(self.completed),
            "attempted": dict(self.attempted), "returned": dict(self.returned),
            "by_phase": deepcopy(self.by_phase), "events": deepcopy(self.events),
            "failed": self.failed, "caps": dict(self.caps),
            "candidate_panel_complete": self.completed == list(self.quotas),
            "scope": "adapter-dispatched action_encoder and predict only",
            "raw_model_bypass_detected": "unknown",
            "predictor_internal_calls": "unknown",
            "upstream_and_other_costs": "unknown",
            "equal_total_compute": False,
            "P3_admission": False, "C3_admission": False,
        }
