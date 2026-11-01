"""Reusable online feedback coordination; no simulator, model, or GPU imports.

See docs/d2p_feedback_controller.md for the callback contract and trust boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class Plan:
    actions: Any
    prediction: Any


@dataclass(frozen=True)
class Execution:
    actual_actions: Any
    rgb: Any
    terminated: bool
    truncated: bool
    error: str | None = None


def compiler_plan(compiler: Any, context: Mapping[str, Any]) -> Plan:
    """Adapt a D2P compiler result to the feedback runtime.

    The complete selected program is retained in ``prediction`` for the journal;
    only its explicit action witness is sent to the executor.  No predicted
    context is installed by this adapter.
    """
    compile_fn = getattr(compiler, "compile", None)
    if not callable(compile_fn):
        raise ValueError("compiler must expose compile(context)")
    result = compile_fn(_context(context))
    if not isinstance(result, Mapping):
        raise ValueError("compiler result must be a mapping")
    programs = result.get("programs")
    selected = result.get("selected_index")
    if (not isinstance(programs, list) or type(selected) is not int
            or not 0 <= selected < len(programs)):
        raise ValueError("compiler result lacks selected program")
    program = programs[selected]
    if not isinstance(program, Mapping):
        raise ValueError("selected program must be a mapping")
    actions = _actions(program.get("actions"))
    if len(actions) == 0:
        raise ValueError("selected program has no action witness")
    return Plan(actions=actions, prediction={
        "schema": result.get("schema"),
        "selected_index": selected,
        "selected_candidate": program.get("candidate"),
        "selected_program": deepcopy(program),
        "operation_counts": deepcopy(result.get("operation_counts")),
        "actual_neural_cost_certified": result.get("actual_neural_cost_certified", False),
    })


def run_compiled_feedback(
    initial_context: Mapping[str, Any],
    *,
    compiler_factory: Callable[[Mapping[str, Any]], Any],
    execute: Callable[[np.ndarray], Execution],
    observe: Callable[[Mapping[str, Any], np.ndarray, np.ndarray], Mapping[str, Any]],
    packet_size: int,
    max_replans: int,
    event_sink: Callable[[dict], None] | None = None,
) -> dict:
    """Run a compiler-backed real-feedback loop.

    ``compiler_factory`` is called only with the current RGB/history context.
    Its returned compiler may use frozen model predictions, but the controller
    installs only the observed context returned by ``observe``.
    """
    if not callable(compiler_factory):
        raise ValueError("compiler_factory must be callable")

    def plan(context):
        return compiler_plan(compiler_factory(_context(context)), context)

    return run_feedback(
        initial_context,
        plan=plan,
        execute=execute,
        observe=observe,
        packet_size=packet_size,
        max_replans=max_replans,
        event_sink=event_sink,
    )


_FORBIDDEN_KEYS = frozenset({
    "reward", "rewards", "state", "states", "env_state", "simulator_state",
    "goal_state", "object_state", "ground_truth", "ground_truth_state",
})


def _normal_key(key: str) -> str:
    return key.casefold().replace("-", "_").replace(" ", "_")


def _context(value):
    """Restrict context containers and reject explicit simulator/reward fields."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = _normal_key(key) if isinstance(key, str) else ""
            forbidden = (
                normalized in _FORBIDDEN_KEYS
                or normalized.endswith("_state")
                or normalized.endswith("_reward")
                or normalized.endswith("_rewards")
            )
            if not isinstance(key, str) or forbidden:
                raise ValueError("context contains forbidden or non-string key")
            result[key] = _context(item)
        return result
    if isinstance(value, (list, tuple)):
        items = [_context(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("context array must be finite numeric or boolean")
        return value.copy()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real) and np.isfinite(value):
        return value
    raise ValueError("unsupported context value")


def _actions(value, *, allow_empty=False):
    try:
        array = np.asarray(value)
        if (array.ndim != 2 or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all() or array.shape[1] == 0
                or (array.shape[0] == 0 and not allow_empty)):
            raise ValueError("actions must be finite numeric [steps, dimensions]")
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid actions") from exc
    return array.copy()


def _rgb(value):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("RGB must be uint8 HxWx3")
    return image.copy()


def run_feedback(
    initial_context: Mapping[str, Any],
    *,
    plan: Callable[[Mapping[str, Any]], Plan],
    execute: Callable[[np.ndarray], Execution],
    observe: Callable[[Mapping[str, Any], np.ndarray, np.ndarray], Mapping[str, Any]],
    packet_size: int,
    max_replans: int,
    event_sink: Callable[[dict], None] | None = None,
) -> dict:
    """Plan initially, then at most ``max_replans`` after completed packets."""
    if (not all(callable(f) for f in (plan, execute, observe))
            or type(packet_size) is not int or packet_size <= 0
            or type(max_replans) is not int or max_replans < 0
            or (event_sink is not None and not callable(event_sink))):
        raise ValueError("invalid callbacks or configuration")
    if not isinstance(initial_context, Mapping):
        raise ValueError("context must be a mapping")
    context = _context(initial_context)
    counts = {"plan_attempted": 0, "plan_returned": 0,
              "execute_attempted": 0, "execute_returned": 0,
              "observe_attempted": 0, "observe_returned": 0,
              "packets_complete": 0, "executed_actions_reported": 0}
    events = []
    known = True
    status, error = "failed", None
    stage = "initial"
    sink_failed = False

    def emit(kind, **fields):
        nonlocal sink_failed
        event = {"index": len(events), "kind": kind, **deepcopy(fields)}
        events.append(event)
        if event_sink is not None:
            try:
                event_sink(deepcopy(event))
            except Exception:
                sink_failed = True
                raise

    try:
        for iteration in range(max_replans + 1):
            stage = "plan"
            counts["plan_attempted"] += 1
            emit("plan_attempt", iteration=iteration)
            proposal = plan(_context(context))
            counts["plan_returned"] += 1
            if not isinstance(proposal, Plan):
                raise ValueError("plan must return Plan")
            actions = _actions(proposal.actions)
            if len(actions) < packet_size:
                raise ValueError("plan shorter than packet")
            # Deep copy prediction only for recording: never use it as context.
            emit("plan_return", iteration=iteration, actions=actions,
                 prediction=deepcopy(proposal.prediction))
            packet = actions[:packet_size].copy()
            stage = "execute"
            counts["execute_attempted"] += 1
            emit("execute_attempt", iteration=iteration, packet=packet)
            try:
                response = execute(packet.copy())
            except Exception:
                known = False
                raise
            counts["execute_returned"] += 1
            if not isinstance(response, Execution):
                known = False
                raise ValueError("execute must return Execution")
            try:
                actual = _actions(response.actual_actions, allow_empty=True)
                if actual.shape[1] != packet.shape[1] or len(actual) > packet_size:
                    raise ValueError("actual action shape or count")
            except Exception:
                known = False
                raise
            counts["executed_actions_reported"] += len(actual)
            # Preserve acknowledged actions even if later metadata is invalid.
            emit("actual_actions", iteration=iteration, actions=actual)
            if (type(response.terminated) is not bool or type(response.truncated) is not bool
                    or (response.error is not None and not isinstance(response.error, str))):
                raise ValueError("invalid execution status")
            image = _rgb(response.rgb)
            emit("execute_return", iteration=iteration, rgb=image,
                 terminated=response.terminated, truncated=response.truncated, error=response.error)
            if response.error is not None:
                raise RuntimeError("execution failure: " + response.error)
            complete = len(actual) == packet_size
            if not complete:
                if not (response.terminated or response.truncated):
                    raise ValueError("partial packet without native stop")
                status = "terminated" if response.terminated else "truncated"
                break
            counts["packets_complete"] += 1
            stage = "observe"
            counts["observe_attempted"] += 1
            emit("observe_attempt", iteration=iteration)
            observed = observe(_context(context), actual.copy(), image.copy())
            counts["observe_returned"] += 1
            if not isinstance(observed, Mapping):
                raise ValueError("observe must return a context mapping")
            context = _context(observed)
            emit("observe_return", iteration=iteration, context=_context(context))
            if response.terminated or response.truncated:
                status = "terminated" if response.terminated else "truncated"
                break
        else:
            status = "replan_limit_reached"
    except Exception as exc:
        error = {"stage": "event_sink" if sink_failed else stage,
                 "type": type(exc).__name__, "message": str(exc)[:1000]}
        status = "failed"
    return {"status": status, "error": error, "context": _context(context), "events": events,
            "counts": counts, "executed_action_count_fully_known": known,
            "accounting_scope": "callback attempts/returns and reported actions only; not full compute budget",
            "P1_admission": False}
