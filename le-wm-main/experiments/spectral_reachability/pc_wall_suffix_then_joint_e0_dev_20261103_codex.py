"""Budgeted suffix-first CEM, with stored incumbents and paired prior controls."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_wall_joint_vs_suffix_e0_dev_20261103_codex as prev

base, joint = prev.base, prev.joint
PROTOCOL = prev.diverse.previous.ROOT.parent / "docs/PC_WALL_SUFFIX_THEN_JOINT_E0_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = prev.diverse.previous.ROOT / "results/pc_wall_suffix_then_joint_e0_dev_20261103_codex.json"


def plan(task, index, cap, control):
    started = time.monotonic()
    starts = np.asarray(control["starts"])
    scores = np.asarray(control["start_scores"])
    ix = int(np.argmin(scores))
    best, best_cost = starts[ix].copy(), float(scores[ix])
    incumbent = {"point": best.tolist(), "distance": best_cost}
    calls = 33 + 8 * 11
    remaining = cap - calls
    allocations = [remaining // 16 + (i < remaining % 16) for i in range(16)]
    rng = np.random.default_rng(2026112500 + index)
    trace = []
    for i, start in enumerate(starts):
        mean, std = start.copy(), np.full(4, .5)
        for iteration in range(2):
            n = allocations[2 * i + iteration]
            samples = np.clip(mean + std * rng.standard_normal((n, 4)), -1, 1)
            if iteration == 0:
                samples[:, :2] = start[:2]
            trajectories, _ = base.rollout(task["start"], joint.decode(samples))
            calls += n
            costs = np.linalg.norm(trajectories[:, -1, :2] - task["goal"], axis=1)
            order = np.argsort(costs, kind="stable")
            if costs[order[0]] < best_cost:
                best, best_cost = samples[order[0]].copy(), float(costs[order[0]])
            elite = samples[order[:max(2, n // 8)]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
            if iteration == 0:
                std[:2] = .1
            trace.append({"start": i, "round": iteration, "parameters": samples.tolist(), "scores": costs.tolist()})
    action = joint.decode(best[None])
    trajectory, blocked = base.rollout(task["start"], action)
    final = float(np.linalg.norm(trajectory[0, -1, :2] - task["goal"]))
    if calls != cap or not np.isclose(final, best_cost, atol=1e-12, rtol=0):
        raise ValueError("replay or budget mismatch")
    return {"rollout_calls": calls, "incumbent": incumbent, "selected": best.tolist(),
            "actions": action[0].tolist(), "distance": final, "trajectory": trajectory[0].tolist(),
            "blocked": blocked[0].tolist(), "trace": trace, "seconds": time.monotonic() - started}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(prev.OUTPUT.read_text())
    if any(base.digest(Path(path)) != value for path, value in source["dependencies"].items()):
        raise ValueError("prior result dependency drift")
    paths = (PROTOCOL, Path(__file__), prev.OUTPUT, Path(prev.__file__), Path(joint.__file__),
             Path(base.__file__), Path(base.physics.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    print("Post-outcome development: suffix first, then restricted joint refinement.", flush=True)
    for row in source["rows"]:
        results = []
        for cap in prev.diverse.previous.BUDGETS:
            fixed = next(p for p in row["plans"] if p["rollout_calls"] == cap and p["method"] == "fixed")
            joint_control = next(p for p in row["plans"] if p["rollout_calls"] == cap and p["method"] == "joint")
            result = plan(row["task"], row["index"], cap, fixed)
            results.append({"cap": cap, "result": result,
                            "fixed_control": fixed["distance"], "joint_control": joint_control["distance"]})
        rows.append({"index": row["index"], "task": row["task"], "results": results})
        print(f"task {row['index']} complete", flush=True)
    if any(base.digest(Path(p)) != h for p, h in hashes.items()):
        raise ValueError("dependency drift")
    summary = {}
    for cap in prev.diverse.previous.BUDGETS:
        data = [r for row in rows for r in row["results"] if r["cap"] == cap]
        summary[str(cap)] = {name: {"mean": float(np.mean(vals)), "successes": sum(x < .05 for x in vals)}
                             for name, vals in (("fixed", [r["fixed_control"] for r in data]),
                                                ("joint", [r["joint_control"] for r in data]),
                                                ("suffix_then_joint", [r["result"]["distance"] for r in data]))}
    with OUTPUT.open("x") as f:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary, "PC2_pass": False,
                   "PC3_pass": False, "full_cost_certified": False}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
