"""Full-horizon response initialization on fixed exact wall tasks."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import pc_exact_wall_kill as physics
from . import d2p_goal_mode_rank_20261102 as modes

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_WALL_RESPONSE_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_wall_response_side_20261103.json"
ARMS = ("task", "gain", "ridge", "random", "flat", "primitive")
CAPS = (512, 1024)
HIGH = np.array([.3, np.pi / 2])
HORIZON = 16


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]} for y, theta, gy in
            itertools.product((-.25, .25), (-np.pi / 4, np.pi / 4), (-.25, .25))]


def rollout(start, actions):
    """Batch exact backend, including full trajectories and blocked events."""
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 3 or actions.shape[1:] != (16, 2)
            or not np.isfinite(actions).all() or np.any(np.abs(actions) > HIGH + 1e-12)):
        raise ValueError("invalid action batch")
    state = np.broadcast_to(start, (len(actions), 3)).copy()
    trace, blocked = [state.copy()], []
    for t in range(16):
        a = actions[:, t]
        displacement = a[:, :1] * np.stack((np.cos(state[:, 2]), np.sin(state[:, 2])), axis=1)
        state[:, :2], event = physics.step_batch(state[:, :2], displacement)
        state[:, 2] += a[:, 1]
        trace.append(state.copy())
        blocked.append(event)
    return np.stack(trace, axis=1), np.stack(blocked, axis=1)


def decode(parameters, primitive=False):
    parameters = np.asarray(parameters)
    if not primitive:
        return parameters.reshape(-1, 16, 2) * HIGH
    out = np.zeros((len(parameters), 16, 2))
    out[:, 0, 1] = parameters[:, 0] * HIGH[1]
    out[:, 1:6, 0] = parameters[:, 1, None] * HIGH[0]
    out[:, 6:8, 1] = parameters[:, 2, None] * HIGH[1]
    out[:, 8:13, 0] = parameters[:, 3, None] * HIGH[0]
    return out


class Evaluator:
    def __init__(self, task, cap):
        self.task, self.cap, self.calls = task, cap, 0

    def endpoints(self, points, primitive=False):
        if self.calls + len(points) > self.cap:
            raise ValueError("budget exhausted")
        self.calls += len(points)
        trace, _ = rollout(self.task["start"], decode(points, primitive))
        return trace[:, -1, :2]

    def scores(self, points, primitive=False):
        return np.linalg.norm(self.endpoints(points, primitive) - self.task["goal"], axis=1)


def plan(task, index, cap, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    started = time.monotonic()
    evaluator = Evaluator(task, cap)
    primitive = arm == "primitive"
    nominals = np.random.default_rng(2026111400 + index).uniform(-.6, .6, (4, 32))
    matrices, svd, solves = [], 0, 0
    if arm in ("task", "gain", "ridge"):
        starts = []
        for nominal in nominals:
            endpoint = evaluator.endpoints(nominal[None])[0]
            epsilon = 1e-5
            positive = evaluator.endpoints(nominal + epsilon * np.eye(32))
            negative = evaluator.endpoints(nominal - epsilon * np.eye(32))
            matrix = ((positive - negative) / (2 * epsilon)).T
            residual = np.asarray(task["goal"]) - endpoint
            if arm == "ridge":
                delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + .001 * np.eye(2), residual)
                solves += 1
            else:
                ranked = modes.rank_goal_modes(matrix, residual, .001)
                choice = int(ranked["order"][0]) if arm == "task" else 0
                delta = ranked["right_t"][choice] * ranked["coefficients"][choice]
                svd += 1
            delta *= min(1., .5 / max(float(np.max(np.abs(delta))), 1e-30))
            starts.append(np.clip(nominal + delta, -1., 1.))
            matrices.append(matrix.tolist())
        starts = np.asarray(starts)
    elif arm == "random":
        starts = nominals
    else:
        starts = np.zeros((4, 4 if primitive else 32))
    costs = evaluator.scores(starts, primitive)
    generation_calls = evaluator.calls
    chosen = int(np.argmin(costs))
    best, best_cost = starts[chosen].copy(), float(costs[chosen])
    remaining = cap - evaluator.calls
    allocations = [remaining // 16 + (i < remaining % 16) for i in range(16)]
    rng = np.random.default_rng(2026111500 + index)
    optimization_trace = []
    for start_index, start in enumerate(starts):
        mean, std = start.copy(), np.full(start.shape, .5)
        for iteration in range(4):
            size = allocations[4 * start_index + iteration]
            samples = np.clip(mean + std * rng.standard_normal((size, len(mean))), -1., 1.)
            values = evaluator.scores(samples, primitive)
            order = np.argsort(values, kind="stable")
            if values[order[0]] < best_cost:
                best, best_cost = samples[order[0]].copy(), float(values[order[0]])
            elites = samples[order[:max(2, size // 8)]]
            mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), .05)
            optimization_trace.append({"start": start_index, "round": iteration,
                                       "samples": size, "best_so_far": best_cost})
    elapsed = time.monotonic() - started
    selected = decode(best[None], primitive)
    trace, blocked = rollout(task["start"], selected)
    replay_error = float(np.linalg.norm(trace[0, -1, :2] - task["goal"]))
    if abs(replay_error - best_cost) > 1e-12 or evaluator.calls != cap:
        raise AssertionError("replay or budget mismatch")
    return {"arm": arm, "cap": cap, "actions": selected[0].tolist(),
            "trajectory": trace[0].tolist(), "blocked": blocked[0].tolist(),
            "terminal_distance": best_cost, "initial_distance": float(np.linalg.norm(
                np.asarray(task["start"])[:2] - task["goal"])),
            "prefix_distance": float(np.linalg.norm(trace[0, 4, :2] - task["goal"])),
            "starts": starts.tolist(), "start_costs": costs.tolist(),
            "responses": matrices, "generation_calls": generation_calls,
            "total_calls": evaluator.calls, "model_transitions": evaluator.calls * 16,
            "svd_calls": svd, "solve_calls": solves, "planning_seconds": elapsed,
            "allocations": allocations, "optimization_trace": optimization_trace}


def feasibility(task):
    """Privileged diagnostic, never an initializer or selection signal."""
    records = []
    for height in (-1.1, 1.1):
        position, heading = np.array(task["start"][:2]), task["start"][2]
        actions = []
        for target in (np.array([0., height]), np.array(task["goal"])):
            diff = target - position
            angle = np.arctan2(diff[1], diff[0])
            turn = (angle - heading + np.pi) % (2 * np.pi) - np.pi
            nt = int(np.ceil(abs(turn) / HIGH[1]))
            if nt:
                actions.extend([[0., turn / nt]] * nt)
            nm = int(np.ceil(np.linalg.norm(diff) / HIGH[0]))
            actions.extend([[float(np.linalg.norm(diff) / nm), 0.]] * nm)
            position, heading = target, angle
        if len(actions) > 16:
            records.append({"height": height, "steps_needed": len(actions), "within_horizon": False})
            continue
        steps = len(actions)
        actions += [[0., 0.]] * (16 - steps)
        trace, blocked = rollout(task["start"], [actions])
        records.append({"height": height, "steps_needed": steps, "within_horizon": True,
                        "actions": actions, "terminal_distance": float(np.linalg.norm(
                            trace[0, -1, :2] - task["goal"])), "blocked": blocked[0].tolist()})
    return records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [PROTOCOL, Path(__file__), Path(physics.__file__), Path(modes.__file__)]
    hashes = {str(p): digest(p) for p in paths}
    print("Fixed exact wall task family; no RGB or P1 admission.", flush=True)
    rows = []
    for index, task in enumerate(tasks()):
        results = [plan(task, index, cap, arm) for cap in CAPS for arm in ARMS]
        rows.append({"index": index, "task": task, "results": results,
                     "feasibility_only": feasibility(task)})
        print(f"completed task {index}", flush=True)
    summary = {str(cap): {arm: {
        "mean_distance": float(np.mean([r["terminal_distance"] for row in rows
                                       for r in row["results"] if r["arm"] == arm and r["cap"] == cap])),
        "successes_below_005": sum(r["terminal_distance"] < .05 for row in rows
                                  for r in row["results"] if r["arm"] == arm and r["cap"] == cap)}
        for arm in ARMS} for cap in CAPS}
    if hashes != {str(p): digest(p) for p in paths}:
        raise RuntimeError("source changed during execution")
    with OUTPUT.open("x") as handle:
        json.dump({"dependencies": hashes, "scope": "exposed exact development family",
                   "P1_admission": False, "full_cost_certified": False,
                   "rows": rows, "summary": summary}, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
