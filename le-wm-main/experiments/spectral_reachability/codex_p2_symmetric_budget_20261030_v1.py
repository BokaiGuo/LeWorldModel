"""Exploratory symmetric-work planner kernel; no production backend is provided.

The caller must meter actual backend operations, persist failures, and bind inputs.
This module's returned counts are requested work, not hardware measurements.
"""
from __future__ import annotations

import time
import numpy as np

ARMS = ("joint", "suffix")
ALPHAS = (0.25, 0.5, 1.0, 2.0)
WALL_SECONDS = 450.0
EXPECTED = {"predict": 39, "jacobian": 2, "vjp_rows_requested": 384,
            "jacobian_elements_requested": 11520, "svd": 2}


def plan(backend, context, goal, arm, clock=time.monotonic):
    """Run two fixed updates; backend rollout returns (three contexts, cost vector).

    jacobian returns the 192x30 terminal Jacobian and must internally perform
    exactly three predict calls. The contract cannot authenticate a backend.
    """
    if arm not in ARMS:
        raise ValueError("unknown arm")
    goal = np.asarray(goal, dtype=np.float64)
    if goal.shape != (192,) or not np.isfinite(goal).all():
        raise ValueError("invalid goal")
    start = clock()
    last = start
    counts = dict.fromkeys(EXPECTED, 0)
    events = []
    result = {"arm": arm, "counts_requested": counts, "events": events,
              "status": "partial", "winner": None, "export": None}

    def dispatch(kind, amount, callback):
        nonlocal last
        now = clock()
        if not np.isfinite(now) or now < last:
            raise ValueError("invalid clock")
        last = now
        if now - start >= WALL_SECONDS:
            raise TimeoutError("wall cap before dispatch")
        for key, n in amount.items():
            if counts[key] + n > EXPECTED[key]:
                raise ValueError("resource cap")
        for key, n in amount.items():
            counts[key] += n
        events.append({"kind": kind, "state": "attempt", "elapsed": now - start})
        value = callback()
        now = clock()
        if not np.isfinite(now) or now < last:
            raise ValueError("invalid clock")
        last = now
        events.append({"kind": kind, "state": "return", "elapsed": now - start})
        if now - start > WALL_SECONDS:
            raise TimeoutError("in-flight wall overrun")
        return value

    def rollout(u):
        contexts, terminal = dispatch("rollout", {"predict": 3},
                                     lambda: backend.rollout(context, u.copy()))
        terminal = np.asarray(terminal, dtype=np.float64)
        if len(contexts) != 3 or terminal.shape != (192,) or not np.isfinite(terminal).all():
            raise ValueError("invalid rollout")
        return contexts, terminal

    def candidate(u):
        contexts, terminal = rollout(u)
        return {"logits": u.copy(), "contexts": contexts, "terminal": terminal,
                "cost": float(np.mean((terminal-goal)**2))}

    try:
        winner = candidate(np.zeros(30))
        result["winner"] = winner
        for iteration in range(2):
            u = winner["logits"].copy()
            j = dispatch("jacobian", {"predict": 3, "jacobian": 1,
                          "vjp_rows_requested": 192, "jacobian_elements_requested": 5760},
                         lambda: backend.jacobian(context, u.copy()))
            j = np.asarray(j, dtype=np.float64)
            if j.shape != (192, 30) or not np.isfinite(j).all():
                raise ValueError("invalid Jacobian")
            mask = np.ones(30)
            if arm == "suffix":
                mask[:10 if iteration == 0 else 20] = 0
            restricted = j * mask
            left, singular, right = dispatch("svd", {"svd": 1},
                                             lambda: np.linalg.svd(restricted, full_matrices=False))
            step = right.T @ ((singular / (singular**2 + 0.01)) *
                              (left.T @ (goal - winner["terminal"])))
            step *= mask
            norm = np.linalg.norm(step)
            if norm > 1.0:
                step /= norm
            for alpha in ALPHAS:
                proposal = candidate(np.clip(u + alpha * step, -3.0, 3.0))
                if proposal["cost"] < winner["cost"]:
                    winner = proposal
            result["winner"] = winner
        # Two separately paid complete replays; no free reconstruction.
        replay1, terminal1 = rollout(winner["logits"])
        replay2, terminal2 = rollout(winner["logits"])
        if not np.array_equal(terminal1, winner["terminal"]) or not np.array_equal(terminal2, terminal1):
            raise ValueError("replay mismatch")
        result["export"] = {"contexts": replay1, "second_contexts": replay2}
        result["status"] = "complete"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_seconds"] = last-start
    result["equal_total_compute"] = False
    result["flops"] = None
    return result
