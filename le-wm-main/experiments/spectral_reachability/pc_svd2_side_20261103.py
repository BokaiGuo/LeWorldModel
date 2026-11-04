"""Two-dimensional response diagnostic; exact dynamics, development only."""
import json
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_spectral_start_diagnostic_20261103_codex as prior

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_SVD2_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_svd2_side_20261103.json"
METHODS = ("svd2", "svd1_same_matrix", "ridge_no_svd", "old_turn_only", "fixed", "random")
RIDGE = 1e-4


def endpoint(velocities, turn):
    state = prior.base.step(np.zeros(3), [velocities[0], turn])
    return prior.base.step(state, [velocities[1], 0.])[:2]


def response_matrix(turn, epsilon=1e-5):
    columns = []
    for direction in np.eye(2):
        columns.append((endpoint(epsilon * direction, turn)
                        - endpoint(-epsilon * direction, turn)) / (2 * epsilon))
    return np.stack(columns, axis=1)


def project(matrix, goal, method):
    if method == "ridge_no_svd":
        return np.linalg.solve(matrix.T @ matrix + RIDGE * np.eye(2), matrix.T @ goal)
    if method not in ("svd2", "svd1_same_matrix"):
        raise ValueError("unknown projection method")
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    gain = singular / (singular**2 + RIDGE)
    if method == "svd1_same_matrix":
        gain[1:] = 0.
    return vh.T @ (gain * (u.T @ goal))


def bank(goal, method):
    if method == "old_turn_only":
        return prior.response_bank(goal, True)
    entries = []
    for turn in np.linspace(-prior.base.TURN, prior.base.TURN, 9):
        matrix = response_matrix(turn)
        velocities = np.clip(project(matrix, goal, method), -1., 1.)
        point = [float(velocities[0]), float(turn), float(velocities[1])]
        error, _ = prior.base.evaluate(prior.chain(point), goal)
        entries.append({"point": point, "error": error, "response": matrix.tolist()})
    return entries


def diagnose(point, goal):
    actions = prior.chain(point)
    error, trace = prior.base.evaluate(actions, goal)
    prefix_error = float(np.linalg.norm(trace[1, :2] - goal))
    same_error, _ = prior.base.evaluate(np.array([[0., 0.], actions[1]]), goal)
    zero_opt_error, _ = prior.base.evaluate(
        np.array([[0., 0.], [np.clip(goal[0], -1., 1.), 0.]]), goal)
    initial = float(np.linalg.norm(goal))
    # With zero velocity, this exact model has no passive translation.
    active_selected = error - prefix_error
    active_zero = same_error - initial
    nonprogress = prefix_error >= initial - 1e-12
    return {"replay_error": error, "trace": trace.tolist(), "prefix_error": prefix_error,
            "initial_error": initial, "nonprogress": bool(nonprogress),
            "zero_prefix_same_suffix_error": same_error,
            "zero_prefix_optimal_suffix_error": zero_opt_error,
            "active_selected_minus_zero": active_selected - active_zero,
            "limited_zero_control_witness": bool(
                nonprogress and active_selected < active_zero - 1e-12
                and error < zero_opt_error - 1e-12),
            "diagnostic_replay_transitions": 6}


def run_one(goal, index, cap, method):
    if method not in METHODS:
        raise ValueError("unknown arm")
    started = time.monotonic()
    entries = []
    if method in METHODS[:4]:
        entries = bank(goal, method)
        starts = [x["point"] for x in sorted(entries, key=lambda x: x["error"])[:4]]
    elif method == "fixed":
        starts = [[0., 0., 0.], [0., prior.base.TURN/2, .5],
                  [0., -prior.base.TURN/2, .5], [0., 0., -.5]]
    else:
        starts = np.random.default_rng(2026110900 + index).uniform(
            [-1., -prior.base.TURN, -1.], [1., prior.base.TURN, 1.], (4, 3)).tolist()
    result = prior.optimize(goal, starts, cap, len(entries))
    optimizer_error = result["error"]
    if entries:
        best = min(entries, key=lambda x: x["error"])
        if best["error"] < result["error"]:
            result["point"], result["error"] = best["point"], best["error"]
    new_response = method in METHODS[:3]
    result.update({
        "method": method, "starts": starts, "candidates": entries,
        "optimizer_error": optimizer_error,
        "generation_best_error": min(x["error"] for x in entries) if entries else None,
        "finite_difference_chain_calls": 36 if new_response else 0,
        "finite_difference_transitions": 72 if new_response else (18 if entries else 0),
        "generation_dynamics_transitions": 90 if new_response else (45 if entries else 0),
        "svd_calls": 9 if method in ("svd2", "svd1_same_matrix", "old_turn_only") else 0,
        "linear_solve_calls": 9 if method == "ridge_no_svd" else 0,
        "planning_seconds": time.monotonic() - started})
    result["diagnostic"] = diagnose(result["point"], goal)
    if not np.isclose(result["error"], result["diagnostic"]["replay_error"], rtol=1e-10, atol=1e-12):
        raise AssertionError("chosen plan differs from replay")
    return result


def summarize(rows):
    output = {}
    for cap in (16, 32, 64):
        output[str(cap)] = {}
        for method in METHODS:
            items = [x for row in rows for x in row["results"]
                     if x["cap"] == cap and x["method"] == method]
            output[str(cap)][method] = {
                "mean_error": float(np.mean([x["error"] for x in items])),
                "max_error": max(x["error"] for x in items),
                "mean_actual_objectives": float(np.mean([x["total_objective_evaluations"] for x in items])),
                "nonprogress": sum(x["diagnostic"]["nonprogress"] for x in items),
                "limited_zero_control_witnesses": sum(
                    x["diagnostic"]["limited_zero_control_witness"] for x in items)}
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("2x2 response SVD vs rank-one and no-SVD controls; 16 exposed exact roots.", flush=True)
    rows = []
    for index in range(16):
        angle = 2 * np.pi * (index + .37) / 16
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        results = [run_one(goal, index, cap, method)
                   for cap in (16, 32, 64) for method in METHODS]
        rows.append({"index": index, "goal": goal.tolist(), "results": results})
        print(f"completed root {index}", flush=True)
    report = {
        "scope": "exposed exact-dynamics development only; no P1 or PC gate admission",
        "full_cost_certified": False, "ridge": RIDGE,
        "sha256": {str(path): prior.sha(path) for path in (
            PROTOCOL, Path(__file__), Path(prior.__file__), Path(prior.base.__file__),
            Path(prior.strong.__file__))},
        "rows": rows, "summary": summarize(rows)}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
