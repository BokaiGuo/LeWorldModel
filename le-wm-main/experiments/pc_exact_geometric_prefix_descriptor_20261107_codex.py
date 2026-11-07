"""Fixed-opportunity action-effect retention using exposed geometric chain proposals."""
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as base
from experiments import pc_exact_goal_relative_bridge_20261107_codex as geom

PROTOCOL = geom.ROOT.parent / "docs/PC_EXACT_GEOMETRIC_PREFIX_DESCRIPTOR_PROTOCOL_20261107_CODEX.md"
OUTPUT = geom.ROOT / "results/pc_exact_geometric_prefix_descriptor_20261107_codex.json"
ARMS = ("parent", "immediate", "lookahead", "descriptor", "shuffled")


def select_with_incumbent(pool, parent_cost, refined, indices):
    """Only selected refinements compete against a common pool incumbent."""
    pool = np.asarray(pool)
    costs = np.asarray(parent_cost, dtype=float)
    if pool.shape != (8, 15, 2) or costs.shape != (8,) or len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("invalid candidate pool or retained slots")
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= 8 for i in indices):
        raise ValueError("invalid retained index")
    if not np.isfinite(costs).all():
        raise ValueError("nonfinite costs")
    incumbent = int(np.argmin(costs))
    best = {"source": "incumbent", "prefix": incumbent, "score": float(costs[incumbent]),
            "chain": pool[incumbent].tolist()}
    for i in indices:
        result = refined[i]
        index = int(result["best"])
        score = float(result["costs"][index])
        if not np.isfinite(score):
            raise ValueError("nonfinite refinement cost")
        if score < best["score"]:
            best = {"source": "refinement", "prefix": i, "score": score, "chain": result["actions"][index]}
    return best


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(geom.OUTPUT.read_text())
    dependencies = {str(p): base.sha(p) for p in (
        Path(__file__), PROTOCOL, geom.OUTPUT, Path(geom.__file__), Path(base.__file__),
        Path(base.exact.__file__), Path(base.exact.wall.__file__), Path(base.exact.wall.physics.__file__))}
    if len(source["rows"]) != 24:
        raise ValueError("expected fixed 24-root panel")
    rows = []
    print("Fixed geometric prefixes: compare lookahead and action-effect retention without B leakage.", flush=True)
    for root, row in enumerate(source["rows"]):
        started = time.perf_counter()
        if row["root"] != root:
            raise ValueError("root mismatch")
        task = row["task"]
        all_chains = np.asarray(row["actions"], dtype=np.float64)
        if all_chains.shape != (384, 15, 2) or np.max(np.abs(all_chains)) > 1 + 1e-12:
            raise ValueError("source candidate mismatch")
        ids = np.random.default_rng(2717000 + root).choice(384, size=24, replace=False)
        pool, a, b = all_chains[ids[:8]], all_chains[ids[8:16], 5:], all_chains[ids[16:], 5:]
        goal = np.asarray(task["goal"], dtype=float)
        costs = np.linalg.norm(base.exact.endpoints(task, pool) - goal, axis=1)
        np.testing.assert_allclose(costs, np.asarray(row["costs"])[ids[:8]], atol=1e-10, rtol=0)
        immediate = np.linalg.norm(base.exact.endpoints(task, pool[:, :5]) - goal, axis=1)

        def cross(tails):
            chained = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, len(tails), 5, 2)),
                                      np.broadcast_to(tails[None], (8, len(tails), 10, 2))), axis=2)
            return np.linalg.norm(base.exact.endpoints(task, chained.reshape(-1, 15, 2)) - goal, axis=1).reshape(8, -1)

        hold = cross(np.zeros((1, 10, 2)))[:, 0]
        a_cost, b_cost = cross(a), cross(b)
        # select() centers rows, so using costs is exactly equivalent to hold-minus-cost
        # for the diversity distance; lookahead is computed from the costs themselves.
        selected = base.select(a_cost, costs, immediate, 2718000 + root)
        refined = [base.optimize(task, chain[:5], chain[5:], 36, 2719000 + 8 * root + i)
                   for i, chain in enumerate(pool)]
        outcome = {arm: select_with_incumbent(pool, costs, refined, selected[arm]) for arm in ARMS}
        predictions = base.heldout_predictions(a, b, a_cost, costs)
        mse = {name: float(np.square(value - b_cost).mean()) for name, value in predictions.items()}
        rows.append({"root": root, "task": task, "source_indices": ids.tolist(), "parent_cost": costs.tolist(),
                     "immediate": immediate.tolist(), "hold_cost": hold.tolist(), "a_cost": a_cost.tolist(),
                     "effect": (hold[:, None] - a_cost).tolist(), "b_cost": b_cost.tolist(),
                     "heldout_mse": mse, "selected": selected, "refined": refined, "outcomes": outcome,
                     "deployment_dynamics_steps": 8 * 15 + 8 * 5 + 8 * 15 + 64 * 15 + 72 * 15,
                     "evaluation_only_dynamics_steps": 64 * 15 + 6 * 36 * 15,
                     "elapsed_including_evaluation_seconds": time.perf_counter() - started})
        print(f"root {root}: lookahead={outcome['lookahead']['score']:.5f}, "
              f"descriptor={outcome['descriptor']['score']:.5f}", flush=True)
    if dependencies != {path: base.sha(path) for path in dependencies}:
        raise ValueError("dependency changed during run")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact-state development; not RGB or independent confirmation"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
