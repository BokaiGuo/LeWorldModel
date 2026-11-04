"""Post-pilot random bridge and finite-difference controls for exact wall."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.spectral_reachability import pc_exact_wall_kill as wall


def bridge_program(bridge):
    points = np.stack((wall.START, np.asarray(bridge, dtype=float), wall.GOAL))
    first = np.linalg.norm(points[1] - points[0])
    second = np.linalg.norm(points[2] - points[1])
    total = first + second
    samples = []
    for distance in np.linspace(0.0, total, wall.HORIZON + 1):
        if distance <= first and first > 0:
            samples.append(points[0] + distance / first * (points[1] - points[0]))
        elif second > 0:
            samples.append(points[1] + (distance - first) / second * (points[2] - points[1]))
        else:
            samples.append(points[-1])
    return wall.project(np.diff(samples, axis=0))


def random_bridge():
    bridges = np.random.default_rng(20261103).uniform([-0.5, -1.5], [0.5, 1.5], size=(64, 2))
    programs = np.stack([bridge_program(p) for p in bridges])
    costs, blocked = wall.rollout_batch(programs)
    best = int(np.argmin(costs))
    return {"seed": 20261103, "selected_index": best, "best_cost": float(costs[best]),
            "selected_program": programs[best].tolist(),
            "candidates": [{"bridge": p.tolist(), "cost": float(c), "blocked": int(b)}
                           for p, c, b in zip(bridges, costs, blocked)],
            "ledger": {"scores": 64, "transitions": 64 * wall.HORIZON}}


def gradient():
    rng = np.random.default_rng(20261104)
    rows = []
    for start_index in range(4):
        start = rng.uniform(-wall.RADIUS, wall.RADIUS, size=2 * wall.HORIZON)
        scores = []
        def objective(flat):
            program = wall.project(np.asarray(flat).reshape(wall.HORIZON, 2))
            result, _ = wall.rollout_batch(program[None])
            scores.append(float(result[0]))
            return scores[-1]
        result = minimize(objective, start, method="L-BFGS-B",
                          bounds=[(-wall.RADIUS, wall.RADIUS)] * len(start),
                          options={"maxiter": 30})
        program = wall.project(result.x.reshape(wall.HORIZON, 2))
        # On abnormal line-search termination SciPy's fun need not match x.
        # Rank the executable returned program by an explicitly counted replay.
        replay_cost = objective(result.x)
        rows.append({"start_index": start_index, "initial_parameters": start.tolist(),
                     "selected_program": program.tolist(), "cost": replay_cost,
                     "optimizer_reported_fun": float(result.fun),
                     "iterations": int(result.nit), "status": int(result.status),
                     "message": str(result.message), "objective_scores": scores})
    best = int(np.argmin([r["cost"] for r in rows]))
    count = sum(len(r["objective_scores"]) for r in rows)
    return {"seed": 20261104, "selected_start": best, "best_cost": rows[best]["cost"],
            "starts": rows, "ledger": {"scores": count, "transitions": count * wall.HORIZON}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = {"random_bridge": random_bridge(), "gradient": gradient()}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "post-pilot exact wall development controls, not independent",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parent_dynamics_sha256": hashlib.sha256(Path(wall.__file__).read_bytes()).hexdigest(),
              "results": results}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"best_cost": v["best_cost"], "ledger": v["ledger"]}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
