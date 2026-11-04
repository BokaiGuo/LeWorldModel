"""Fail-stop routing for conditional compilation and common model hooks.

Not a host runner: this object neither authenticates the model nor takes a GPU
lock. The caller owns the underlying durable ledger and all runtime safeguards.
"""
from contextlib import contextmanager
from copy import deepcopy
import threading

from . import p2_same_bank_native_objective_20261102_exclusive as native

NAME = "p2_conditional_hook_meter_20261102_exclusive"
HOOKS = ("action_encoder", "predict", "predictor", "pred_proj")
COMPILE = {
    "conditional_prefix_transition": 16,
    "conditional_suffix_transition": 32,
    "conditional_jacobian_192_output": 16,
    "conditional_bounded_ridge_svd_solve": 16,
    **{axis: 48 for axis in HOOKS},
}
COMMON = {"transition_packet": 288, "autograd_grad": 128,
          **{axis: 288 for axis in HOOKS}}
CAPS = {"composition_inherited": COMPILE,
        **{"common_" + arm: dict(COMMON) for arm in native.optimizer.ARMS}}
HOOK_CATEGORIES = frozenset(CAPS)
OP_CATEGORIES = {
    "composition_inherited": frozenset(COMPILE).difference(HOOKS),
    **{"common_" + arm: frozenset(("transition_packet", "autograd_grad"))
       for arm in native.optimizer.ARMS},
}


class _LedgerView:
    """Ledger subset used by the unmodified conditional bridge."""
    def __init__(self, router):
        self._router = router

    @property
    def category(self):
        return self._router._ledger.category

    @contextmanager
    def phase(self, category):
        self._router._thread()
        try:
            with self._router._ledger.phase(category):
                yield
        except BaseException as exc:
            self._router.failed = True
            self._router._ledger.record("routed_phase_failed", {
                "requested_category": str(category), "error_type": type(exc).__name__})
            raise

    def record(self, kind, payload):
        self._router._thread()
        return self._router._ledger.record(kind, payload)

    def call(self, axis, operation):
        return self._router._dispatch("operation", axis, operation)


class RoutedModelMeter:
    """Supply `ledger` to bridge and this instance to model_scope.

    Every dispatch is durably attempted by the underlying ledger before the
    operation. A rejection or failure latches the meter; later dispatches and
    completion fail closed. Single-threaded use only.
    """
    def __init__(self, ledger):
        if not all(callable(getattr(ledger, name, None)) for name in ("record", "call", "phase")):
            raise ValueError("durable ledger interface required")
        if getattr(ledger, "category", None) is not None:
            raise ValueError("ledger must begin outside phase")
        self._ledger = ledger
        self._owner = threading.get_ident()
        self.failed = False
        self._finished = False
        self._attempted = {category: dict.fromkeys(axes, 0) for category, axes in CAPS.items()}
        self._returned = deepcopy(self._attempted)
        self.ledger = _LedgerView(self)
        self.ledger_view = self.ledger
        self._ledger.record("routed_meter_open", {"caps": CAPS, "single_thread_only": True})

    @property
    def counts(self):
        """Snapshot of attempted counts, grouped by complete ledger category."""
        return deepcopy(self._attempted)

    @property
    def returned_counts(self):
        return deepcopy(self._returned)

    def phase(self, category):
        return self.ledger.phase(category)

    def _thread(self):
        if threading.get_ident() != self._owner:
            self.failed = True
            raise RuntimeError("routed meter requires its owner thread")

    def _reject(self, reason, axis, category, route):
        self.failed = True
        self._ledger.record("routed_meter_rejected", {"reason": reason, "axis": str(axis),
                                                     "category": category, "route": route})
        raise ValueError(reason)

    def _dispatch(self, route, axis, operation):
        self._thread()
        category = self._ledger.category
        if self.failed or self._finished:
            self._reject("meter already failed or finished", axis, category, route)
        allowed = (HOOKS if route == "hook" and category in HOOK_CATEGORIES
                   else OP_CATEGORIES.get(category, ()) if route == "operation" else ())
        if not isinstance(axis, str) or axis not in allowed or not callable(operation):
            self._reject("invalid route/category/axis/operation", axis, category, route)
        if self._attempted[category][axis] >= CAPS[category][axis]:
            self._reject("attempt cap exceeded", axis, category, route)
        self._attempted[category][axis] += 1
        try:
            result = self._ledger.call(axis, operation)
        except BaseException:
            self.failed = True
            raise
        if self.failed:
            self._ledger.record("routed_meter_nested_failure",
                                {"category": category, "axis": axis, "route": route})
            raise RuntimeError("nested dispatch failed despite caught exception")
        self._returned[category][axis] += 1
        return result

    def call(self, axis, operation):
        """Route model hooks and metered bridge operations by axis."""
        return self._dispatch("hook" if axis in HOOKS else "operation", axis, operation)

    def finish(self):
        """Require a full successful schedule; incomplete work is not refunded."""
        self._thread()
        if self.failed or self._finished or self._ledger.category is not None or self._attempted != CAPS or self._returned != CAPS:
            self.failed = True
            self._ledger.record("routed_meter_incomplete", {"attempted": self._attempted,
                                                            "returned": self._returned})
            raise ValueError("failed or incomplete routed schedule")
        self._ledger.record("routed_meter_complete", {"attempted": self._attempted,
                                                       "returned": self._returned})
        self._finished = True
        return {"attempted": deepcopy(self._attempted), "returned": deepcopy(self._returned),
                "equal_total_compute": False, "native_execution_authenticated": False}


RoutedMeter = RoutedModelMeter
