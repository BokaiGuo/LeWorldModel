"""Post-seal plan injection and matched-optimizer initialization diagnostics."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.spectral_reachability import pc_exact_unicycle_development as base
from experiments.spectral_reachability import pc_exact_unicycle_strong_baseline as strong

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_SPECTRAL_START_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md"
SOURCE = ROOT / "results/pc_exact_orientation_mechanism_20261103_codex.json"
AUDIT = ROOT / "results/pc_exact_orientation_selected_audit_20261103_codex.json"
OUTPUT = ROOT / "results/pc_spectral_start_diagnostic_20261103_codex.json"
BOUNDS = [(-1., 1.), (-base.TURN, base.TURN), (-1., 1.)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(point):
    return np.array([[point[0], point[1]], [point[2], 0.]])


def response_bank(goal, use_svd):
    entries = []
    for angle in np.linspace(-base.TURN, base.TURN, 9):
        successor = base.step(np.zeros(3), [0., angle])
        plus = base.step(successor, [1e-5, 0.])
        minus = base.step(successor, [-1e-5, 0.])
        response = (plus[:2] - minus[:2]) / 2e-5
        norm2 = response @ response
        if norm2 <= 1e-20:
            raise ValueError("degenerate response")
        direction = float(np.linalg.svd(response.reshape(2, 1), full_matrices=False)[2][0, 0]) if use_svd else 1.
        velocity = np.clip(np.dot(goal - successor[:2], response * direction) / norm2, -1., 1.) * direction
        point = np.array([0., angle, velocity])
        score, _ = base.evaluate(chain(point), goal)
        entries.append({"point": point.tolist(), "error": score})
    return entries


def optimize(goal, starts, budget, generation_evaluations):
    available = budget - generation_evaluations
    if available < len(starts):
        raise ValueError("budget insufficient for four starts")
    allocation = [available // len(starts) + (i < available % len(starts)) for i in range(len(starts))]
    best = None
    best_squared = float("inf")
    count = 0
    for start, limit in zip(starts, allocation, strict=True):
        local_count = 0

        def evaluate(point):
            nonlocal count, local_count, best, best_squared
            if local_count >= limit:
                raise strong.BudgetExhausted
            value, grad = strong.objective_gradient(point, goal)
            count += 1
            local_count += 1
            if value < best_squared:
                best, best_squared = np.array(point), value
            return value, grad

        try:
            minimize(evaluate, start, method="L-BFGS-B", jac=True, bounds=BOUNDS,
                     options={"maxfun": limit, "maxiter": limit, "ftol": 1e-12})
        except strong.BudgetExhausted:
            pass
    return {"point": best.tolist(), "error": float(np.sqrt(best_squared)), "allocations": allocation,
            "optimizer_objective_evaluations": count, "analytic_gradient_evaluations": count,
            "generation_objective_evaluations": generation_evaluations,
            "total_objective_evaluations": count + generation_evaluations, "cap": budget}


def run(source, audit):
    if audit["source_sha256"] != sha(SOURCE):
        raise ValueError("audit-source binding mismatch")
    indices = [row["index"] for row in audit["rows"] if row["witness"]]
    if indices != [3, 4, 11, 12]:
        raise ValueError("unexpected sealed witness selection")
    output = []
    for index in indices:
        root = next(row for row in source["rows"] if row["index"] == index)
        goal = np.asarray(root["goal"])
        injected = np.asarray(root["combined_actions"])
        injection_error, _ = base.evaluate(injected, goal)
        baseline_bank = response_bank(goal, True)
        scores = [x["error"] for x in baseline_bank] + [injection_error]
        rank = list(np.argsort(scores, kind="stable")).index(9) + 1
        selected_prefix = base.step(np.zeros(3), injected[0])
        direction = np.array([np.cos(selected_prefix[2]), np.sin(selected_prefix[2])])
        velocity = np.clip(np.dot(goal - selected_prefix[:2], direction), -1., 1.)
        suffix_opt_error, _ = base.evaluate(np.stack([injected[0], [velocity, 0.]]), goal)
        injection = {"chain": injected.tolist(), "error": injection_error, "rank_of_ten": rank,
                     "nine_bank_errors": scores[:9], "nonzero_first_velocity": bool(abs(injected[0, 0]) > 1e-12),
                     "minimum_prefix_action_l2": float(min(np.linalg.norm(injected[0] - chain(x["point"])[0])
                                                           for x in baseline_bank)),
                     "gradient_prefix_optimized_suffix_error": suffix_opt_error,
                     "intermediate_pruning": "none in restricted bank"}
        results = []
        for budget in (16, 32, 64):
            for method in ("spectral", "response_no_svd", "fixed", "random"):
                start_time = time.monotonic()
                if method in ("spectral", "response_no_svd"):
                    bank = response_bank(goal, method == "spectral")
                    ordered = sorted(bank, key=lambda x: x["error"])
                    starts = [x["point"] for x in ordered[:4]]
                    generation_evaluations = 9
                elif method == "fixed":
                    starts = [[0., 0., 0.], [0., base.TURN/2, .5],
                              [0., -base.TURN/2, .5], [0., 0., -.5]]
                    generation_evaluations = 0
                else:
                    rng = np.random.default_rng(2026110800 + index)
                    starts = rng.uniform([-1., -base.TURN, -1.], [1., base.TURN, 1.], (4, 3)).tolist()
                    generation_evaluations = 0
                outcome = optimize(goal, starts, budget, generation_evaluations)
                outcome.update({"method": method, "starts": starts, "budget": budget,
                                "finite_difference_transitions": 18 if generation_evaluations else 0,
                                "generation_dynamics_transitions": 45 if generation_evaluations else 0,
                                "svd_calls": 9 if method == "spectral" else 0,
                                "wall_seconds": time.monotonic() - start_time})
                results.append(outcome)
        output.append({"root": index, "goal": goal.tolist(), "injection": injection, "methods": results})
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Exact-dynamics post-seal injection and initialization diagnostics; no RGB claim.", flush=True)
    source, audit = json.loads(SOURCE.read_text()), json.loads(AUDIT.read_text())
    rows = run(source, audit)
    report = {"source_sha256": sha(SOURCE), "audit_sha256": sha(AUDIT), "protocol_sha256": sha(PROTOCOL),
              "runner_sha256": sha(Path(__file__)), "base_sha256": sha(Path(base.__file__)),
              "gradient_sha256": sha(Path(strong.__file__)),
              "scope": "selected exact-dynamics development roots; no confirmation", "rows": rows}
    with OUTPUT.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"roots": len(rows), "injection_ranks": [r["injection"]["rank_of_ten"] for r in rows]}))


if __name__ == "__main__":
    main()
