"""CPU-only scheduling kernel. The host must meter all model operations.

This module neither imports a model nor executes physical actions. Model-derived
scores supplied by the caller are not evidence that a chunk is physically safe.
"""
from dataclasses import dataclass
import math
from typing import Callable


AXES = ("predict", "predictor_samples", "encode", "ad", "svd")


def _integer(value):
    return type(value) is int and value >= 0


def vector(value):
    if type(value) is not dict or set(value) != set(AXES) or not all(
        _integer(value[k]) for k in AXES
    ):
        raise ValueError("complete nonnegative integer resource vector required")
    return dict(value)


class Meter:
    """Non-refundable attempts, including errors; return counts separate."""

    def __init__(self, caps):
        self.caps = vector(caps)
        self.attempted = dict.fromkeys(AXES, 0)
        self.returned = dict.fromkeys(AXES, 0)
        self.failed = False
        self.events = []

    def call(self, phase, cost, operation):
        charge = vector(cost)
        if self.failed or not callable(operation):
            raise ValueError("meter closed or invalid operation")
        if any(self.attempted[k] + charge[k] > self.caps[k] for k in AXES):
            raise ValueError("resource cap exceeded before dispatch")
        for k in AXES:
            self.attempted[k] += charge[k]
        event = {"phase": phase, "charge": charge.copy(), "returned": False}
        self.events.append(event)
        try:
            result = operation()
        except BaseException:
            self.failed = True
            raise
        for k in AXES:
            self.returned[k] += charge[k]
        event["returned"] = True
        return result


@dataclass(frozen=True)
class Config:
    base_steps: int
    extension_steps: int
    max_extensions: int
    min_predicted_improvement: float
    max_model_disagreement: float

    def __post_init__(self):
        if (not _integer(self.base_steps) or self.base_steps == 0 or
            not _integer(self.extension_steps) or self.extension_steps == 0 or
            not _integer(self.max_extensions)):
            raise ValueError("invalid chunk dimensions")
        for value in (self.min_predicted_improvement, self.max_model_disagreement):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid fixed threshold")


@dataclass(frozen=True)
class Signal:
    """Scores computed before execution of the proposed extension."""
    current_model_cost: float
    extended_model_cost: float
    model_disagreement: float

    def valid(self):
        return all(type(x) in (int, float) and math.isfinite(x) for x in (
            self.current_model_cost, self.extended_model_cost, self.model_disagreement
        )) and self.model_disagreement >= 0


def choose(config: Config, signal: Signal):
    if not isinstance(signal, Signal) or not signal.valid():
        return False, "invalid_signal"
    if signal.model_disagreement > config.max_model_disagreement:
        return False, "disagreement"
    if signal.current_model_cost - signal.extended_model_cost < config.min_predicted_improvement:
        return False, "insufficient_predicted_improvement"
    return True, "extend"


def schedule(config: Config, arm: str, meter: Meter, signal_cost: dict,
             acquire_signal: Callable[[int, int], Signal]):
    """Return a chunk length before action execution; never receives outcomes.

    The caller must bind the callback to the current model context and selected
    action sequence, and truthfully instrument its resource charge. A callback
    runs once per proposed extension; rejected work remains charged.
    """
    if arm not in ("fixed", "dynamic") or not isinstance(meter, Meter):
        raise ValueError("invalid arm or meter")
    charge = vector(signal_cost)
    if arm == "dynamic" and not any(charge.values()):
        raise ValueError("dynamic signal must have an explicit nonzero charge")
    length = config.base_steps
    decisions = []
    if arm == "dynamic":
        for index in range(config.max_extensions):
            signal = meter.call("extension_signal", charge,
                                lambda: acquire_signal(length, length + config.extension_steps))
            accepted, reason = choose(config, signal)
            decisions.append({"index": index, "from_steps": length,
                              "to_steps": length + config.extension_steps,
                              "accepted": accepted, "reason": reason})
            if not accepted:
                break
            length += config.extension_steps
    return {"arm": arm, "chunk_steps": length, "decisions": decisions,
            "attempted": meter.attempted.copy(), "returned": meter.returned.copy(),
            "caps": meter.caps.copy(), "P3_admission": False,
            "equal_total_compute": False, "physical_safety_certified": False}
