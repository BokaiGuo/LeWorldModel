"""Outcome-sealed exact-dynamics response program development experiment."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.spectral_reachability import pc_exact_unicycle_development as dynamics
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_PROGRAM_JOINT_REFINEMENT_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_response_program_joint_refinement_20261103_codex.json"
BOUNDS = [(-1., 1.), (-dynamics.TURN, dynamics.TURN), (-1., 1.)]
METHODS = ("response", "same_bank_greedy", "random_bank", "flat")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actions(point):
    return np.array([[point[0], point[1]], [point[2], 0.]])


def endpoint(velocity, turn):
    state = dynamics.step(np.zeros(3), [velocity[0], turn])
    return dynamics.step(state, [velocity[1], 0.])[:2]


def response_bank(goal):
    result = []
    for turn in np.linspace(-dynamics.TURN, dynamics.TURN, 9):
        eps = 1e-5
        columns = [(endpoint(eps * unit, turn) - endpoint(-eps * unit, turn)) / (2 * eps)
                   for unit in np.eye(2)]
        matrix = np.stack(columns, axis=1)
        velocity = np.clip(np.linalg.solve(matrix.T @ matrix + 1e-4 * np.eye(2),
                                          matrix.T @ goal), -1., 1.)
        point = np.array([velocity[0], turn, velocity[1]])
        score, _ = dynamics.evaluate(actions(point), goal)
        result.append({"point": point.tolist(), "terminal_distance": score})
    return result


def random_bank(goal, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-1., -dynamics.TURN, -1.], [1., dynamics.TURN, 1.], (9, 3))
    return [{"point": point.tolist(), "terminal_distance": dynamics.evaluate(actions(point), goal)[0]}
            for point in points]


def refine(goal, starts, cap, generation_calls, incumbent=None):
    available = cap - generation_calls
    if len(starts) != 4 or available < 4:
        raise ValueError("four starts and positive allocations required")
    allocations = [available // 4 + (i < available % 4) for i in range(4)]
    if incumbent is None:
        best, best_squared = None, float("inf")
    else:
        best = np.asarray(incumbent["point"], dtype=float).copy()
        best_squared = float(incumbent["terminal_distance"]) ** 2
    evaluated, actual = 0, []
    for start, limit in zip(starts, allocations, strict=True):
        local = 0

        def evaluate(point):
            nonlocal best, best_squared, evaluated, local
            if local >= limit:
                raise strong.BudgetExhausted
            value, gradient = strong.objective_gradient(point, goal)
            evaluated += 1
            local += 1
            if value < best_squared:
                best, best_squared = np.array(point).copy(), float(value)
            return value, gradient

        try:
            minimize(evaluate, start, method="L-BFGS-B", jac=True, bounds=BOUNDS,
                     options={"maxfun": limit, "maxiter": limit, "ftol": 1e-12})
        except strong.BudgetExhausted:
            pass
        actual.append(local)
    return {"point": best.tolist(), "terminal_distance": float(np.sqrt(best_squared)),
            "pre_refinement_incumbent": incumbent, "allocations": allocations,
            "actual_per_start": actual, "generation_calls": generation_calls,
            "optimizer_objectives": evaluated, "analytic_gradients": evaluated,
            "total_calls": generation_calls + evaluated, "cap": cap}


def plan(goal, index, cap, method):
    start_time = time.monotonic()
    if method in ("response", "same_bank_greedy"):
        bank = response_bank(goal)
        fd_calls, solves, prefix_scores = 36, 9, 0
        if method == "same_bank_greedy":
            for item in bank:
                prefix = dynamics.step(np.zeros(3), actions(item["point"])[0])
                item["prefix_distance"] = float(np.linalg.norm(prefix[:2] - goal))
            prefix_scores = 9
            selected = sorted(bank, key=lambda item: item["prefix_distance"])[:4]
        else:
            selected = sorted(bank, key=lambda item: item["terminal_distance"])[:4]
        generation_calls = fd_calls + len(bank) + prefix_scores
    elif method == "random_bank":
        bank = random_bank(goal, 2026111000 + index)
        selected = sorted(bank, key=lambda item: item["terminal_distance"])[:4]
        fd_calls, solves, prefix_scores = 0, 0, 0
        generation_calls = 9
    elif method == "flat":
        rng = np.random.default_rng(2026111100 + index)
        selected = [{"point": x.tolist()} for x in rng.uniform(
            [-1., -dynamics.TURN, -1.], [1., dynamics.TURN, 1.], (4, 3))]
        bank, fd_calls, solves, prefix_scores, generation_calls = [], 0, 0, 0, 0
    else:
        raise ValueError("unknown method")
    incumbent = min(selected, key=lambda item: item["terminal_distance"]) if bank else None
    result = refine(goal, [x["point"] for x in selected], cap, generation_calls, incumbent)
    result.update({"method": method, "starts": selected, "bank": bank,
                   "finite_difference_chain_calls": fd_calls, "ridge_solves": solves,
                   "prefix_score_calls": prefix_scores,
                   "generation_dynamics_steps": 2 * (fd_calls + len(bank)) + prefix_scores,
                   "planning_seconds": time.monotonic() - start_time})
    return result


def intervention(goal, selected, greedy):
    selected_actions, greedy_actions = actions(selected), actions(greedy)
    suffix = selected_actions[1]
    initial = float(np.linalg.norm(goal))
    records = {}
    for label, prefix in (("selected", selected_actions[0]), ("zero", np.zeros(2)),
                          ("same_bank_greedy", greedy_actions[0])):
        state = dynamics.step(np.zeros(3), prefix)
        prefix_distance = float(np.linalg.norm(state[:2] - goal))
        final, _ = dynamics.evaluate(np.stack([prefix, suffix]), goal)
        response = np.array([np.cos(state[2]), np.sin(state[2])])
        velocity = np.clip(np.dot(goal - state[:2], response), -1., 1.)
        optimal, _ = dynamics.evaluate(np.stack([prefix, [velocity, 0.]]), goal)
        records[label] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                          "common_suffix_distance": final, "active_change": final - prefix_distance,
                          "optimal_suffix_distance": optimal}
    selected_record = records["selected"]
    return {"initial_distance": initial, "common_suffix": suffix.tolist(), "controls": records,
            "limited_witness": bool(
                selected_record["prefix_distance"] >= initial - 1e-12
                and all(selected_record["active_change"] < records[k]["active_change"] - 1e-12
                        and selected_record["optimal_suffix_distance"] < records[k]["optimal_suffix_distance"] - 1e-12
                        for k in ("zero", "same_bank_greedy")))}


def run():
    rows = []
    for radius in (.65, .95):
        for j in range(8):
            index = len(rows)
            theta = 2 * np.pi * (j + .238) / 8
            goal = radius * np.array([np.cos(theta), np.sin(theta)])
            results = []
            for cap in (64, 128, 256):
                plans = [plan(goal, index, cap, method) for method in METHODS]
                plans[0]["intervention"] = intervention(goal, plans[0]["point"], plans[1]["point"])
                results.extend(plans)
            rows.append({"index": index, "goal": goal.tolist(), "results": results})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Exact-dynamics response programs with joint refinement; not RGB or confirmation.", flush=True)
    rows = run()
    summary = {}
    for cap in (64, 128, 256):
        summary[str(cap)] = {
            method: {"mean_error": float(np.mean([next(result["terminal_distance"] for result in row["results"]
                                                      if result["method"] == method and result["cap"] == cap)
                                                   for row in rows])),
                     "mean_total_calls": float(np.mean([next(result["total_calls"] for result in row["results"]
                                                            if result["method"] == method and result["cap"] == cap)
                                                       for row in rows]))}
            for method in METHODS}
        summary[str(cap)]["limited_witnesses"] = sum(
            next(result["intervention"]["limited_witness"] for result in row["results"]
                 if result["method"] == "response" and result["cap"] == cap) for row in rows)
    report = {"scope": "exact-dynamics development; no gate passage or RGB claim",
              "protocol_sha256": digest(PROTOCOL), "runner_sha256": digest(Path(__file__)),
              "dynamics_sha256": digest(Path(dynamics.__file__)), "gradient_sha256": digest(Path(strong.__file__)),
              "summary": summary, "rows": rows}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
