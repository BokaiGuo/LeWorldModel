"""Analytical witness and bounded multistart gradient on sealed unicycle roots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.spectral_reachability import pc_exact_unicycle_development as base


def objective_gradient(parameters, goal):
    v0, turn, v1 = np.asarray(parameters, dtype=np.float64)
    c, s = np.cos(turn), np.sin(turn)
    residual = np.array([v0 + v1 * c, v1 * s]) - np.asarray(goal)
    jac = np.array([[1.0, -v1 * s, c], [0.0, v1 * c, s]])
    return float(residual @ residual), 2.0 * jac.T @ residual


def analytical_witness(goal):
    goal = np.asarray(goal, dtype=np.float64)
    if goal.shape != (2,) or np.max(np.abs(goal)) > 1 + 1e-12:
        raise ValueError("goal outside the witness contract")
    turn = base.TURN if goal[1] >= 0 else -base.TURN
    return np.array([[goal[0], turn], [abs(goal[1]), 0.0]])


class BudgetExhausted(Exception):
    pass


def multistart_gradient(goal, budget=32):
    if budget < 4:
        raise ValueError("budget must cover fixed starts")
    starts = (
        np.array([0., 0., 0.]),
        np.array([0., base.TURN / 2, 0.5]),
        np.array([0., -base.TURN / 2, 0.5]),
        np.array([0., 0., -0.5]),
    )
    bounds = [(-1., 1.), (-base.TURN, base.TURN), (-1., 1.)]
    count = 0
    best_value = float("inf")
    best_point = None
    per_start_limit = budget // len(starts)
    for start in starts:
        local_count = 0

        def evaluate(point):
            nonlocal count, local_count, best_value, best_point
            if count >= budget or local_count >= per_start_limit:
                raise BudgetExhausted
            value, grad = objective_gradient(point, goal)
            count += 1
            local_count += 1
            if value < best_value:
                best_value, best_point = value, np.array(point, copy=True)
            return value, grad

        try:
            minimize(evaluate, start, method="L-BFGS-B", jac=True, bounds=bounds,
                     options={"maxfun": per_start_limit, "maxiter": per_start_limit, "ftol": 1e-12})
        except BudgetExhausted:
            pass
    v0, turn, v1 = best_point
    return np.array([[v0, turn], [v1, 0.0]]), {
        "objective_evaluations": count, "analytic_gradient_evaluations": count,
        "starts": len(starts), "global_budget": budget, "per_start_limit": per_start_limit,
        "planner_dynamics_transitions": 0,
        "note": "closed-form exact objective, not a black-box dynamics transition count"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source.read_text())
    if len(source["rows"]) != 64:
        raise ValueError("expected sealed 64-direction panel")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in source["rows"]:
        goal = np.asarray(row["goal"], dtype=np.float64)
        gradient, ledger = multistart_gradient(goal)
        oracle = analytical_witness(goal)
        gradient_cost, _ = base.evaluate(gradient, goal)
        oracle_cost, _ = base.evaluate(oracle, goal)
        if oracle_cost > 1e-12:
            raise RuntimeError("analytical witness did not reach the goal")
        rows.append({"index": row["index"], "goal": row["goal"],
                     "original_structured_error": row["methods"]["structured"]["final_error"],
                     "gradient_error": gradient_cost, "gradient_actions": gradient.tolist(),
                     "gradient_ledger": ledger, "oracle_error": oracle_cost,
                     "oracle_actions": oracle.tolist()})
    structured = np.array([row["original_structured_error"] for row in rows])
    gradient = np.array([row["gradient_error"] for row in rows])
    tolerance = 1e-8
    report = {"scope": "known synthetic control only; not learned model, physical task or PC admission",
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": {"count": len(rows), "structured_mean": float(structured.mean()),
                          "gradient_mean": float(gradient.mean()),
                          "oracle_max_error": max(row["oracle_error"] for row in rows),
                          "structured_wins": int(np.sum(structured < gradient - tolerance)),
                          "ties": int(np.sum(np.abs(structured - gradient) <= tolerance)),
                          "gradient_wins": int(np.sum(gradient < structured - tolerance))},
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
