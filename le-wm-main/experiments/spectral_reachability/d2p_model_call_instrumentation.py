"""Metadata-only, synchronous outer-runner instrumentation of model.predict.

No torch import, device allocation, inference, tensor copy/detach, or backward
hook. CUDA device metadata is mandatory for BOTH inputs, including evaluation;
there is no CPU fallback switch. Tests use non-neural metadata fakes.

Each call requires an explicit planning/evaluation phase. Mark physical
feedback/evaluator work as evaluation: never include it in planning counters.
Use a separate phase per root/method and retain external frozen root bindings.
Scopes may share a method label; records are not silently pooled across roots.

Counts are actual Python invocations and sum(input batch sizes), NOT candidate
rollouts or FLOPs. context_lengths is an invocation histogram of observed
embedding history lengths. horizon_tokens is caller-declared scope metadata,
not inferred from history length, and NOT a count of completed rollouts.
Use separate scopes when horizons change. A failed delegated call is counted
as attempted work and separately flagged; its enclosing receipt is incomplete.
Rejected input/unscoped calls never reach predict and consume no predict count.

Only calls through this model instance's predict attribute are intercepted.
Previously cached callables, delegated inner models, encoder work, backward
work, concurrent callers, and compiled/captured execution are outside scope.
Do not mutate model.predict or use instrumentation concurrently while active.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import signature


class PredictInstrumentation:
    """Single-use context manager; restores the exact original attribute state."""

    def __init__(self, model):
        self.model = model
        self._active = False
        self._used = False
        self._current = None
        self._records = []

    @property
    def records(self):
        """Detached JSON-serializable receipts, planning/evaluation kept separate."""
        return deepcopy(self._records)

    def __enter__(self):
        if self._used:
            raise RuntimeError("instrumentation is single-use")
        original = self.model.predict
        if getattr(original, "_d2p_predict_instrumentation", False):
            raise RuntimeError("nested instrumentation on the same model forbidden")
        call_signature = signature(original)
        names = list(call_signature.parameters)
        if len(names) < 2:
            raise TypeError("predict must expose embedding and action input parameters")
        self._had_override = "predict" in vars(self.model)
        self._override = vars(self.model).get("predict")

        @wraps(original)
        def counted(*args, **kwargs):
            if not self._active or self._current is None:
                raise RuntimeError("predict requires an explicit active phase")
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            emb, act = (bound.arguments[name] for name in names[:2])
            batch, context = self._validate_inputs(emb, act)
            receipt = self._current
            receipt["predict_calls"] += 1
            receipt["predict_samples"] += batch
            histogram = receipt["context_lengths"]
            histogram[context] = histogram.get(context, 0) + 1
            try:
                return original(*args, **kwargs)
            except BaseException:
                receipt["failed_predict_calls"] += 1
                raise

        counted._d2p_predict_instrumentation = True
        self.model.predict = counted
        self._used = self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._had_override:
                self.model.predict = self._override
            else:
                delattr(self.model, "predict")
        finally:
            self._active = False
            self._current = None
        return False

    @staticmethod
    def _validate_inputs(emb, act):
        devices = []
        for name, value in (("embedding", emb), ("action", act)):
            device = getattr(value, "device", None)
            if getattr(device, "type", None) != "cuda":
                raise ValueError(f"{name} input must be CUDA; CPU fallback forbidden")
            devices.append(getattr(device, "index", None))
            shape = getattr(value, "shape", ())
            if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError(f"{name} must have positive [batch, context, features] shape")
        if devices[0] != devices[1] or emb.shape[:2] != act.shape[:2]:
            raise ValueError("predict inputs must share CUDA device, batch and context")
        return emb.shape[0], emb.shape[1]

    @contextmanager
    def phase(self, phase, *, method, horizon_tokens):
        """Yield a live receipt; completed remains false on errors/failed calls.

        The yielded dict is for reading after exit; records returns safe copies.
        horizon_tokens must be the declared full imagined action horizon for
        this scope, never the historical context window passed to predict.
        """
        if not self._active or self._current is not None:
            raise RuntimeError("phase requires active instrumentation and cannot nest")
        if (phase not in ("planning", "evaluation") or not isinstance(method, str)
                or not method.strip() or type(horizon_tokens) is not int or horizon_tokens < 1):
            raise ValueError("explicit planning/evaluation, method and positive integer horizon required")
        receipt = {"phase": phase, "method": method, "horizon_tokens": horizon_tokens,
                   "predict_calls": 0, "predict_samples": 0, "context_lengths": {},
                   "failed_predict_calls": 0, "completed": False}
        self._records.append(receipt)
        self._current = receipt
        try:
            yield receipt
            receipt["completed"] = receipt["failed_predict_calls"] == 0
        finally:
            self._current = None
