"""Exact dynamics staged/full-action development comparison."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.spectral_reachability import pc_exact_unicycle_development as base
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_FULL_ACTION_COMPOSITION_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_full_action_composition_20261103_codex.json"
BOUNDS = [(-1., 1.), (-base.TURN, base.TURN), (-1., 1.)]
SCALES = np.array([-2., -1., -.5, -.25, .25, .5, 1., 2.])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_chain(point):
    return np.array([[point[0], point[1]], [point[2], 0.]])


def final_position(point):
    _, trajectory = base.evaluate(make_chain(point), [0., 0.])
    return trajectory[-1, :2]


def generate_response(goal, spectral):
    nominal = np.array([0., 0., .5])
    final = final_position(nominal)
    columns = []
    for i in (0, 1):
        perturb = np.zeros(3)
        perturb[i] = 1e-5
        columns.append((final_position(nominal + perturb) -
                        final_position(nominal - perturb)) / 2e-5)
    jacobian = np.stack(columns, axis=1)
    residual = goal - final
    damping = 1e-4
    if spectral:
        u, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
        step = vh.T @ ((singular / (singular**2 + damping)) * (u.T @ residual))
    else:
        step = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(2),
                               jacobian.T @ residual)
    prefixes = np.clip(SCALES[:, None] * step, [-1., -base.TURN], [1., base.TURN])
    return prefixes, {"nominal": nominal.tolist(), "jacobian": jacobian.tolist(),
                      "residual": residual.tolist(), "step": step.tolist(),
                      "generation_evaluations": 5, "generation_transitions": 10,
                      "svd_calls": int(spectral)}


def optimize(goal, prefixes, budget, generation, staged):
    remaining = budget - generation
    nstarts = len(prefixes)
    if remaining < nstarts:
        raise ValueError("insufficient budget")
    allocation = [remaining // nstarts + (i < remaining % nstarts) for i in range(nstarts)]
    best_value, best = float("inf"), None
    count, actual = 0, []
    for prefix, limit in zip(prefixes, allocation, strict=True):
        prefix = np.asarray(prefix)
        local = 0

        def evaluate(variable):
            nonlocal count, local, best_value, best
            if local >= limit:
                raise strong.BudgetExhausted
            point = np.array([prefix[0], prefix[1], variable[0]]) if staged else np.asarray(variable)
            value, gradient = strong.objective_gradient(point, goal)
            count += 1
            local += 1
            if value < best_value:
                best_value, best = float(value), point.copy()
            return value, gradient[2:] if staged else gradient

        initial = [.5] if staged else [*prefix, .5]
        bounds = BOUNDS[2:] if staged else BOUNDS
        try:
            minimize(evaluate, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                     options={"maxfun": limit, "maxiter": limit, "ftol": 1e-12})
        except strong.BudgetExhausted:
            pass
        actual.append(local)
    return {"point": best.tolist(), "terminal_distance": float(np.sqrt(best_value)),
            "objective_cap": budget, "generation_evaluations": generation,
            "optimizer_evaluations": count, "analytic_gradient_evaluations": count,
            "total_evaluations": generation + count, "unused_evaluations": budget - generation - count,
            "allocations": allocation, "actual_per_start": actual}


def mechanism(goal, point):
    selected = make_chain(point)
    prefix = selected[0]
    suffix = selected[1]
    zero = np.zeros(2)
    initial = float(np.linalg.norm(goal))
    prefix_state = base.step(np.zeros(3), prefix)
    prefix_distance = float(np.linalg.norm(prefix_state[:2] - goal))
    selected_final, _ = base.evaluate(selected, goal)
    zero_same, _ = base.evaluate(np.stack([zero, suffix]), goal)
    direction = np.array([np.cos(prefix_state[2]), np.sin(prefix_state[2])])
    v_selected = np.clip(np.dot(goal - prefix_state[:2], direction), -1., 1.)
    selected_opt, _ = base.evaluate(np.stack([prefix, [v_selected, 0.]]), goal)
    zero_opt, _ = base.evaluate(np.array([[0., 0.], [np.clip(goal[0], -1., 1.), 0.]]), goal)
    return {"initial_distance": initial, "prefix_distance": prefix_distance,
            "selected_final_distance": selected_final, "zero_same_suffix_distance": zero_same,
            "selected_active_change": selected_final - prefix_distance,
            "zero_active_change": zero_same - initial,
            "selected_prefix_optimized_distance": selected_opt, "zero_prefix_optimized_distance": zero_opt}


def run():
    rows = []
    for radius in (.55, .8, 1.05):
        for j in range(8):
            index = len(rows)
            angle = 2 * np.pi * (j + .123) / 8
            goal = radius * np.array([np.cos(angle), np.sin(angle)])
            methods = []
            for budget in (24, 48, 96):
                for name in ("spectral", "response_no_svd", "random", "flat"):
                    started = time.monotonic()
                    if name in ("spectral", "response_no_svd"):
                        prefixes, generation = generate_response(goal, name == "spectral")
                    else:
                        rng = np.random.default_rng(2026110900 + index)
                        prefixes = rng.uniform([-1., -base.TURN], [1., base.TURN], (8, 2))
                        generation = {"generation_evaluations": 0, "generation_transitions": 0, "svd_calls": 0}
                    outcome = optimize(goal, prefixes, budget, generation["generation_evaluations"], name != "flat")
                    outcome.update({"method": name, "budget": budget, "prefixes": prefixes.tolist(),
                                    "generation": generation, "planning_seconds": time.monotonic() - started})
                    if name != "flat":
                        outcome["mechanism_diagnostic"] = mechanism(goal, outcome["point"])
                    methods.append(outcome)
            rows.append({"index": index, "radius": radius, "goal": goal.tolist(), "methods": methods})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Exact full-action staged vs flat search; development and not RGB validation.", flush=True)
    rows = run()
    summary = {}
    passes = 0
    for budget in (24, 48, 96):
        means = {name: float(np.mean([next(x["terminal_distance"] for x in row["methods"]
                                            if x["budget"] == budget and x["method"] == name)
                                     for row in rows]))
                 for name in ("spectral", "response_no_svd", "random", "flat")}
        passed = (means["spectral"] + 1e-9 < means["response_no_svd"]
                  and means["spectral"] + 1e-9 < means["flat"])
        passes += passed
        summary[str(budget)] = {"mean_terminal_distance": means, "strict_development_condition": passed}
    report = {"scope": "synthetic known-dynamics development; not independent confirmation",
              "protocol_sha256": digest(PROTOCOL), "runner_sha256": digest(Path(__file__)),
              "dynamics_sha256": digest(Path(base.__file__)), "gradient_sha256": digest(Path(strong.__file__)),
              "summary": summary, "pause_spectral_construction": passes < 2, "rows": rows}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"summary": summary, "pause_spectral_construction": passes < 2}, indent=2))


if __name__ == "__main__":
    main()
