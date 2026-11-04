"""CEM on the same four-parameter full action template as the program bank."""
import json
import time
from pathlib import Path

import numpy as np

from . import pc_wall_factor_flat_budget_dev_20261103_codex as flat

factor, base = flat.factor, flat.base
PROTOCOL = factor.ROOT.parent / "docs/PC_WALL_FACTOR_JOINT_TEMPLATE_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = factor.ROOT / "results/pc_wall_factor_joint_template_dev_20261103_codex.json"


def decode(parameters):
    p = np.asarray(parameters, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or np.any(np.abs(p) > 1):
        raise ValueError("invalid normalized four-parameter program")
    actions = np.zeros((len(p), 16, 2))
    actions[:, :2, 1] = p[:, 0, None] * np.pi / 2
    actions[:, 2:6, 0] = p[:, 1, None] * .3
    actions[:, 6:8, 1] = p[:, 2, None] * np.pi / 2
    actions[:, 8:, 0] = p[:, 3, None] * .3
    return actions


def plan(task, index, cap):
    started = time.monotonic()
    rng = np.random.default_rng(2026111900 + index)
    best = np.zeros(4)
    trace, _ = base.rollout(task["start"], decode(best[None]))
    best_cost = float(np.linalg.norm(trace[0, -1, :2] - task["goal"]))
    allocations = [(cap - 1) // 16 + (i < (cap - 1) % 16) for i in range(16)]
    steps = []
    for start in range(4):
        mean, std = np.zeros(4), np.full(4, .5)
        for iteration in range(4):
            size = allocations[start * 4 + iteration]
            samples = np.clip(mean + std * rng.standard_normal((size, 4)), -1, 1)
            trajectories, _ = base.rollout(task["start"], decode(samples))
            scores = np.linalg.norm(trajectories[:, -1, :2] - task["goal"], axis=1)
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] < best_cost:
                best, best_cost = samples[order[0]].copy(), float(scores[order[0]])
            elite = samples[order[:max(2, len(order) // 8)]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
            steps.append({"start": start, "iteration": iteration, "samples": samples.tolist(),
                          "scores": scores.tolist(), "best_so_far": best_cost})
    actions = decode(best[None])
    trace, blocked = base.rollout(task["start"], actions)
    if not np.isclose(np.linalg.norm(trace[0, -1, :2] - task["goal"]), best_cost, atol=1e-12):
        raise ValueError("replay mismatch")
    return {"parameters": best.tolist(), "actions": actions[0].tolist(), "trajectory": trace[0].tolist(),
            "blocked": blocked[0].tolist(), "distance": best_cost, "rollout_calls": 1 + sum(allocations),
            "trace": steps, "seconds": time.monotonic() - started}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    old = factor.load_prior()
    factor_report = json.loads(factor.OUTPUT.read_text())
    seal = json.loads(factor.SEAL.read_text())
    if factor_report["seal_sha256"] != base.digest(factor.SEAL) or seal["prior_sha256"] != base.digest(factor.prior.OUTPUT):
        raise ValueError("source binding mismatch")
    paths = (PROTOCOL, Path(__file__), factor.OUTPUT, factor.SEAL, factor.prior.OUTPUT,
             Path(factor.__file__), Path(base.__file__), Path(base.physics.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    print("Same-template joint planning on exposed tasks; development only.", flush=True)
    for row, original in zip(factor_report["rows"], old["rows"], strict=True):
        for suffix_cap, budget in flat.BUDGETS:
            choice = next(b for b in row["budgets"] if b["cap"] == suffix_cap)
            result = plan(original["task"], row["index"], budget)
            programs = {name: choice["arms"][name]["terminal_distance"]
                        for name in ("diverse_original", "diverse_stratified")}
            rows.append({"task": row["index"], "suffix_cap": suffix_cap, "budget": budget,
                         "joint": result, "programs": programs})
        print(f"task {row['index']} complete", flush=True)
    if any(base.digest(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("source changed")
    summary = {}
    for _, budget in flat.BUDGETS:
        subset = [r for r in rows if r["budget"] == budget]
        distances = [r["joint"]["distance"] for r in subset]
        summary[str(budget)] = {"joint_mean": float(np.mean(distances)),
                                "joint_successes": sum(x < .05 for x in distances)}
        for arm in ("diverse_original", "diverse_stratified"):
            values = [r["programs"][arm] for r in subset]
            summary[str(budget)][arm] = {"mean": float(np.mean(values)), "successes": sum(x < .05 for x in values),
                                         "wins_vs_joint": sum(x < y for x, y in zip(values, distances, strict=True))}
    with OUTPUT.open("x") as file:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary, "PC3_pass": False,
                   "full_cost_certified": False}, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
