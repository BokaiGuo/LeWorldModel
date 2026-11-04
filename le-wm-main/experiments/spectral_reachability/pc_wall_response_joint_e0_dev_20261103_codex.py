"""Goal-conditioned response generation and shared joint CEM in exact wall dynamics."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_wall_factor_joint_template_dev_20261103_codex as joint
from .d2p_goal_mode_rank_20261102 import rank_goal_modes

base = joint.base
ROOT = joint.factor.ROOT
PROTOCOL = ROOT.parent / "docs/PC_WALL_RESPONSE_JOINT_E0_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_wall_response_joint_e0_dev_20261103_codex.json"
METHODS = ("goal", "singular", "ridge", "random", "flat")
BUDGETS = (1145, 2169)


def plan(task, index, cap, method):
    if method not in METHODS or cap not in BUDGETS:
        raise ValueError("unregistered method or budget")
    started = time.monotonic()
    calls = 0

    def endpoints(points):
        nonlocal calls
        points = np.asarray(points)
        calls += len(points)
        if calls > cap:
            raise RuntimeError("budget exceeded")
        trace, _ = base.rollout(task["start"], joint.decode(points))
        return trace[:, -1, :2]

    nominal = np.random.default_rng(2026112000 + index).uniform(-.6, .6, (4, 4))
    candidates, info = [], []
    for i, start in enumerate(nominal):
        if method in ("goal", "singular", "ridge"):
            eps = .01
            center = endpoints(start[None])[0]
            plus = endpoints(start + eps * np.eye(4))
            minus = endpoints(start - eps * np.eye(4))
            matrix = ((plus - minus) / (2 * eps)).T
            residual = np.asarray(task["goal"]) - center
            if method == "ridge":
                delta = matrix.T @ np.linalg.solve(matrix @ matrix.T + .001 * np.eye(2), residual)
                evidence = {}
            else:
                ranked = rank_goal_modes(matrix, residual, .001)
                k = int(ranked["order"][0]) if method == "goal" else 0
                delta = ranked["right_t"][k] * ranked["coefficients"][k]
                evidence = {"selected_mode": k, "singular_values": ranked["singular"].tolist(),
                            "goal_contributions": ranked["contributions"].tolist()}
            delta *= min(1., .5 / max(float(np.max(np.abs(delta))), 1e-30))
            candidate = np.clip(start + delta, -1, 1)
            info.append({"nominal": start.tolist(), "endpoint": center.tolist(), "jacobian": matrix.tolist(),
                         "residual": residual.tolist(), **evidence})
        elif method == "random":
            direction = np.random.default_rng(2026112100 + 10 * index + i).normal(size=4)
            candidate = np.clip(start + .5 * direction / np.max(np.abs(direction)), -1, 1)
        else:
            candidate = start.copy()
        candidates.append(candidate)
    candidates = np.asarray(candidates)
    generation_calls = calls
    scores = np.linalg.norm(endpoints(candidates) - task["goal"], axis=1)
    idx = int(np.argmin(scores))
    best, best_score = candidates[idx].copy(), float(scores[idx])
    incumbent = {"parameters": best.tolist(), "distance": best_score}
    remaining = cap - calls
    allocations = [remaining // 16 + (i < remaining % 16) for i in range(16)]
    rng = np.random.default_rng(2026112200 + index)
    trace = []
    for i, start in enumerate(candidates):
        mean, std = start.copy(), np.full(4, .5)
        for iteration in range(4):
            n = allocations[4 * i + iteration]
            samples = np.clip(mean + std * rng.standard_normal((n, 4)), -1, 1)
            costs = np.linalg.norm(endpoints(samples) - task["goal"], axis=1)
            order = np.argsort(costs, kind="stable")
            if costs[order[0]] < best_score:
                best, best_score = samples[order[0]].copy(), float(costs[order[0]])
            elite = samples[order[:max(2, n // 8)]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
            trace.append({"start": i, "round": iteration, "parameters": samples.tolist(),
                          "scores": costs.tolist(), "best_so_far": best_score})
    actions = joint.decode(best[None])
    replay, blocked = base.rollout(task["start"], actions)
    final = float(np.linalg.norm(replay[0, -1, :2] - task["goal"]))
    if calls != cap or not np.isclose(best_score, final, atol=1e-12, rtol=0):
        raise ValueError("budget or selected replay mismatch")
    prefix = actions[0, :6]
    zero_tail = np.zeros((1, 10, 2))
    probes = np.concatenate((joint.factor.prior.probe_family(), actions[:, 6:]), axis=0)
    controls = {}
    for name, pfx in (("selected", prefix), ("zero", np.zeros((6, 2)))):
        values = joint.factor.prior.evaluate(task, pfx, probes)
        zero = joint.factor.prior.evaluate(task, pfx, zero_tail)[0]
        controls[name] = {"prefix": pfx.tolist(), "distances": values.tolist(),
                          "relative_to_zero_suffix": (values - zero).tolist()}
    initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - task["goal"]))
    prefix_distance = float(np.linalg.norm(replay[0, 6, :2] - task["goal"]))
    return {"method": method, "parameters": best.tolist(), "actions": actions[0].tolist(),
            "distance": final, "trajectory": replay[0].tolist(), "blocked": blocked[0].tolist(),
            "initial_distance": initial, "prefix_distance": prefix_distance,
            "nonprogress": bool(prefix_distance >= initial - 1e-12), "common_suffix_controls": controls,
            "candidates": candidates.tolist(), "response_evidence": info, "incumbent": incumbent,
            "generation_calls": generation_calls, "rollout_calls": calls, "svd_calls": 4 if method in ("goal", "singular") else 0,
            "solve_calls": 4 if method == "ridge" else 0, "trace": trace,
            "seconds": time.monotonic() - started}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (PROTOCOL, Path(__file__), Path(joint.__file__), Path(base.__file__), Path(rank_goal_modes.__code__.co_filename),
             Path(base.physics.__file__), Path(joint.factor.prior.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    print("Wall response generation and joint CEM, development only.", flush=True)
    for index, task in enumerate(base.tasks()):
        rows.append({"index": index, "task": task,
                     "plans": [plan(task, index, cap, method) for cap in BUDGETS for method in METHODS]})
        print(f"task {index} complete", flush=True)
    if any(base.digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("dependency changed")
    summary = {}
    for cap in BUDGETS:
        summary[str(cap)] = {}
        for method in METHODS:
            plans = [p for row in rows for p in row["plans"] if p["method"] == method and p["rollout_calls"] == cap]
            summary[str(cap)][method] = {"mean": float(np.mean([p["distance"] for p in plans])),
                                         "successes": sum(p["distance"] < .05 for p in plans),
                                         "nonprogress": sum(p["nonprogress"] for p in plans)}
    with OUTPUT.open("x") as file:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary,
                   "scope": "exposed exact E0 development only",
                   "PC2_pass": False, "PC3_pass": False, "full_cost_certified": False}, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
