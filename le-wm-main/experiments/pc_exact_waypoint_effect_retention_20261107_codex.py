"""Pre-registered exact-wall retention diagnostic on sealed waypoint proposals."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as core
from experiments import pc_exact_goal_relative_bridge_20261107_codex as source

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_EXACT_WAYPOINT_EFFECT_RETENTION_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_waypoint_effect_retention_20261107_codex.json"
ARMS = core.ARMS


def rank_indices(offsets):
    order = np.argsort(offsets, kind="stable")
    return order[np.arange(24, 384, 48)], order[np.arange(12, 384, 48)], order[np.arange(36, 384, 48)]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed = json.loads(source.OUTPUT.read_text())
    if sealed["dependencies"] != {p: core.sha(p) for p in sealed["dependencies"]}:
        raise ValueError("sealed source dependency changed")
    paths = [Path(__file__), PROTOCOL, source.OUTPUT, Path(core.__file__),
             Path(core.exact.__file__), Path(core.exact.wall.__file__),
             Path(core.exact.wall.physics.__file__)]
    dependencies = {str(p): core.sha(p) for p in paths}
    if len(sealed["rows"]) != 24:
        raise ValueError("expected 24 development roots")
    rows = []
    for row in sealed["rows"]:
        root, task = row["root"], row["task"]
        candidates, old_cost = np.asarray(row["actions"]), np.asarray(row["costs"])
        pool_ix, a_ix, b_ix = rank_indices(row["offsets"])
        pool, a, b = candidates[pool_ix], candidates[a_ix, 5:], candidates[b_ix, 5:]
        if len(np.unique(pool[:, :5].reshape(8, -1), axis=0)) != 8:
            raise ValueError(f"root {root}: duplicated prefixes")
        goal = np.asarray(task["goal"])
        parent = old_cost[pool_ix]
        immediate = np.linalg.norm(core.exact.endpoints(task, pool[:, :5]) - goal, axis=1)

        def cross(tails):
            chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, len(tails), 5, 2)),
                                     np.broadcast_to(tails[None], (8, len(tails), 10, 2))), axis=2)
            return np.linalg.norm(core.exact.endpoints(task, chains.reshape(-1, 15, 2)) - goal, axis=1).reshape(8, -1)

        a_cost = cross(a)
        selected = core.select(a_cost, parent, immediate, 2714010 + root)
        # Computed for every prefix solely for paired offline evaluation.
        refined = [core.optimize(task, chain[:5], chain[5:], 36, 2715000 + root * 8 + i)
                   for i, chain in enumerate(pool)]
        incumbent = int(np.argmin(old_cost))
        outcomes = {}
        for arm, indices in selected.items():
            choices = [(float(old_cost[incumbent]), "incumbent", candidates[incumbent])]
            for i in indices:
                result = refined[i]
                j = result["best"]
                choices.append((result["costs"][j], i, result["actions"][j]))
            score, origin, chain = min(choices, key=lambda item: item[0])
            outcomes[arm] = {"score": float(score), "origin": origin, "chain": np.asarray(chain).tolist()}
        b_cost = cross(b)
        predictions = core.heldout_predictions(a, b, a_cost, parent)
        rows.append({"root": root, "task": task, "pool_indices": pool_ix.tolist(),
                     "a_indices": a_ix.tolist(), "b_indices": b_ix.tolist(),
                     "parent": parent.tolist(), "immediate": immediate.tolist(),
                     "a_cost": a_cost.tolist(), "b_cost": b_cost.tolist(), "selected": selected,
                     "refined": refined, "incumbent_index": incumbent, "outcomes": outcomes,
                     "heldout_mse": {name: float(np.square(pred - b_cost).mean())
                                     for name, pred in predictions.items()},
                     "deployment_full_chain_evaluations": 384 + 64 + 72,
                     "deployment_prefix_evaluations": 8})
        print(f"root {root}: " + " ".join(f"{arm}={outcomes[arm]['score']:.4f}" for arm in ARMS), flush=True)
    if dependencies != {p: core.sha(p) for p in dependencies}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact state and dynamics; not RGB-only or independent confirmation"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
