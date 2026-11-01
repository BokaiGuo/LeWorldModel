"""Prospective P3 contracts only; neither stage admission nor compute advantage.

Use one guard per bound root/method for ALL prediction phases, including signal
construction and verification. All predict calls must traverse this attribute;
cached/inner/compiled callables are outside coverage (as in the v1 instrument).
No refunds on exceptions, retries, rejected candidates or unsuccessful episodes.
This cap is prediction samples, NOT encoder/backward/FLOPs/wall-clock budget.
"""
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from math import isfinite
from threading import Lock

from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("positive Python integer required")


@dataclass(frozen=True)
class FrozenSignals:
    """Model-derived pre-outcome scalars; provenance must be sealed externally.

    No rewards, success, physical endpoint, or evaluator fields are accepted.
    A type signature cannot prove that a caller did not leak an outcome.
    """
    source_seal: str
    unreachable_fraction: float
    horizon_gain: float

    def __post_init__(self):
        if not isinstance(self.source_seal, str) or not self.source_seal.strip():
            raise ValueError("frozen signal source seal required")
        for value in (self.unreachable_fraction, self.horizon_gain):
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("finite signal in [0,1] required")


@dataclass(frozen=True)
class FrozenSchedule:
    source_seal: str
    horizons: tuple[int, int]
    base_candidates: int
    expansion_factor: int
    chi_threshold: float
    gain_threshold: float

    def __post_init__(self):
        FrozenSignals(self.source_seal, self.chi_threshold, self.gain_threshold)
        if type(self.horizons) is not tuple or len(self.horizons) != 2:
            raise ValueError("immutable short/long horizon pair required")
        for value in (*self.horizons, self.base_candidates, self.expansion_factor):
            _positive(value)
        if self.horizons[0] >= self.horizons[1]:
            raise ValueError("strictly increasing horizons required")

    def choose(self, signals: FrozenSignals):
        if not isinstance(signals, FrozenSignals) or signals.source_seal != self.source_seal:
            raise ValueError("mixed frozen signal source")
        horizon = self.horizons[int(signals.horizon_gain >= self.gain_threshold)]
        count = self.base_candidates
        if signals.unreachable_fraction >= self.chi_threshold:
            count *= self.expansion_factor
        # Diagnostic lower bound only. Actual prediction work is guarded below;
        # repeated passes / signal formation / verification cost extra.
        return {"horizon_tokens": horizon, "expanded_candidates": count,
                "single_pass_predict_samples": count * horizon}


class PredictionBudgetExceeded(RuntimeError):
    """Rejected before model invocation; previous attempted work stays charged."""


class PredictionBudget(PredictInstrumentation):
    """Single-use CUDA metadata guard layered on existing failed-call accounting.

    Budget debit occurs before delegation, using actual input batch size EACH
    token, not scheduled candidates or successful-only counters. Never reset
    between phases. External runner must retain and verify the bound identity.
    """
    def __init__(self, model, *, binding: Binding, method: str, sample_limit: int):
        super().__init__(model)
        if not isinstance(binding, Binding) or not isinstance(method, str) or not method.strip():
            raise ValueError("explicit binding and method required")
        _positive(sample_limit)
        self._binding, self._method, self._limit = binding, method, sample_limit
        self._spent = self._blocked = 0
        self._lock = Lock()

    def __enter__(self):
        super().__enter__()
        counted = self.model.predict
        call_signature = signature(counted)
        names = list(call_signature.parameters)

        @wraps(counted)
        def guarded(*args, **kwargs):
            if not self._active or self._current is None:
                raise RuntimeError("explicit active phase required")
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            batch, _ = self._validate_inputs(*(bound.arguments[n] for n in names[:2]))
            with self._lock:
                if self._spent + batch > self._limit:
                    self._blocked += 1
                    raise PredictionBudgetExceeded("actual prediction sample cap exhausted")
                self._spent += batch
            return counted(*args, **kwargs)

        self.model.predict = guarded
        return self

    def phase(self, phase, *, method, horizon_tokens):
        if method != self._method:
            raise ValueError("method differs from budget binding")
        return super().phase(phase, method=method, horizon_tokens=horizon_tokens)

    def receipt(self):
        from dataclasses import asdict
        return {"schema": "d2p_adaptive_budget.v2", "binding": asdict(self._binding),
                "method": self._method, "predict_sample_limit": self._limit,
                "attempted_predict_samples": self._spent, "blocked_calls": self._blocked,
                "phases": self.records, "p3_admitted": False, "compute_advantage": False}
