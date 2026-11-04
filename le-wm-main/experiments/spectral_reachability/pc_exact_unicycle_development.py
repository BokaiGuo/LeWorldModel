"""Exact unicycle mechanism pilot; analytical dynamics, no learned models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TURN = np.pi / 2
PREFIXES = (-TURN, -TURN / 2, 0.0, TURN / 2, TURN)


def step(state, action):
    x, y, theta = np.asarray(state, dtype=np.float64)
    velocity, omega = np.asarray(action, dtype=np.float64)
    if abs(velocity) > 1 + 1e-12 or abs(omega) > TURN + 1e-12:
        raise ValueError("action outside bounds")
    return np.array([x + velocity * np.cos(theta), y + velocity * np.sin(theta), theta + omega])


def evaluate(program, goal):
    state = np.zeros(3)
    trajectory = [state.copy()]
    for action in program:
        state = step(state, action)
        trajectory.append(state.copy())
    return float(np.linalg.norm(state[:2] - goal)), np.asarray(trajectory)


def greedy(goal):
    state = np.zeros(3)
    actions = []
    for _ in range(2):
        response = np.array([np.cos(state[2]), np.sin(state[2])])
        v = np.clip(np.dot(goal - state[:2], response), -1, 1)
        action = np.array([v, 0.0])
        state = step(state, action)
        actions.append(action)
    return np.asarray(actions), {"transitions": 2, "objective_evaluations": 0,
                                 "response_evaluations": 2}


def structured(goal):
    candidates = []
    for angle in PREFIXES:
        prefix = np.array([0.0, angle])
        successor = step(np.zeros(3), prefix)
        response = np.array([np.cos(successor[2]), np.sin(successor[2])])
        v = np.clip(np.dot(goal - successor[:2], response), -1, 1)
        suffix = np.array([v, 0.0])
        final = step(successor, suffix)
        cost = float(np.linalg.norm(final[:2] - goal))
        candidates.append((cost, np.stack((prefix, suffix)), response))
    selected_index = int(np.argmin([c[0] for c in candidates]))
    selected = candidates[selected_index]
    zero = candidates[2]
    return selected[1], {"transitions": 10, "objective_evaluations": 5,
                         "response_evaluations": 5, "selected_prefix_index": selected_index,
                         "successor_response": selected[2].tolist(),
                         "zero_prefix_response": zero[2].tolist(),
                         "zero_prefix_final_error": zero[0]}


def search(goal, *, seed, cem):
    rng = np.random.default_rng(seed)
    lower = np.array([-1.0, -TURN, -1.0, -TURN])
    upper = -lower
    mean = np.zeros(4)
    sigma = upper.copy()
    best_cost = float("inf")
    best = None
    for _ in range(4):
        if cem:
            candidates = np.clip(rng.normal(mean, sigma, (8, 4)), lower, upper)
        else:
            candidates = rng.uniform(lower, upper, (8, 4))
        costs = []
        for action in candidates:
            cost, _ = evaluate(action.reshape(2, 2), goal)
            costs.append(cost)
            if cost < best_cost:
                best, best_cost = action.copy().reshape(2, 2), cost
        if cem:
            elite = candidates[np.argsort(costs, kind="stable")[:2]]
            mean = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), 1e-3)
    return best, {"transitions": 64, "objective_evaluations": 32, "response_evaluations": 0}


def run_panel(count=64):
    rows = []
    for index in range(count):
        angle = 2 * np.pi * index / count + 0.031
        goal = np.array([np.cos(angle), np.sin(angle)])
        programs = {
            "greedy": greedy(goal),
            "random": search(goal, seed=72100 + index, cem=False),
            "cem": search(goal, seed=82100 + index, cem=True),
            "structured": structured(goal),
        }
        methods = {}
        for name, (actions, ledger) in programs.items():
            final_cost, trajectory = evaluate(actions, goal)  # diagnostic replay, excluded from planner ledger
            methods[name] = {"actions": actions.tolist(), "initial_error": 1.0,
                             "first_error": float(np.linalg.norm(trajectory[1, :2] - goal)),
                             "final_error": final_cost, "trajectory": trajectory.tolist(), "ledger": ledger}
        structured_row = methods["structured"]
        ledger = structured_row["ledger"]
        initial_projection = abs(float(goal[0]))
        successor_projection = abs(float(np.dot(goal, ledger["successor_response"])))
        rows.append({"index": index, "goal": goal.tolist(), "methods": methods,
                     "mechanism_witness": bool(
                         abs(structured_row["first_error"] - 1.0) < 1e-12
                         and successor_projection > initial_projection + 1e-12
                         and structured_row["final_error"] < ledger["zero_prefix_final_error"] - 1e-12
                     )})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = run_panel()
    summary = {}
    for method in ("greedy", "random", "cem", "structured"):
        values = np.array([r["methods"][method]["final_error"] for r in rows])
        delta = np.array([r["methods"]["structured"]["final_error"] for r in rows]) - values
        summary[method] = {"mean_final_error": float(values.mean()),
                           "structured_wins": int(np.sum(delta < -1e-12)),
                           "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                           "structured_losses": int(np.sum(delta > 1e-12))}
    report = {"scope": "synthetic analytical known dynamics only; not frozen-model or physical validation",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "roots": len(rows), "mechanism_witness_roots": sum(r["mechanism_witness"] for r in rows),
              "diagnostic_replay_excluded_from_planner_ledgers": True, "summary": summary, "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"mechanism_witness_roots": report["mechanism_witness_roots"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
