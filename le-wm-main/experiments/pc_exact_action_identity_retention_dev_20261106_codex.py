"""Fixed-pool action-identity retention kill test on exposed exact wall tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_block_program_wall_prospective_dev_20261103_codex as panel
from experiments import pc_successor_response_composition_e0_dev_20261103_codex as exact

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_EXACT_ACTION_IDENTITY_RETENTION_DEV_PROTOCOL_20261106_CODEX.md"
OUTPUT = ROOT / "results/pc_exact_action_identity_retention_dev_20261106_codex.json"
ARMS = ("parent", "immediate", "lookahead", "descriptor", "shuffled")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(scores, parent, immediate, seed):
    scores = np.asarray(scores)
    look = scores.min(axis=1)
    first = int(np.argmin(look))
    eligible = np.argsort(look, kind="stable")[:6]
    eligible = eligible[eligible != first]
    centered = scores - scores.mean(axis=1, keepdims=True)
    random = np.random.default_rng(seed)
    shuffled = np.stack([row[random.permutation(len(row))] for row in centered])

    def diverse(vectors):
        distances = np.linalg.norm(vectors[eligible] - vectors[first], axis=1)
        return [first, int(eligible[np.argmax(distances)])]

    return {
        "parent": np.argsort(parent, kind="stable")[:2].tolist(),
        "immediate": np.argsort(immediate, kind="stable")[:2].tolist(),
        "lookahead": np.argsort(look, kind="stable")[:2].tolist(),
        "descriptor": diverse(centered),
        "shuffled": diverse(shuffled),
    }


def optimize(task, prefix, initial_tail, samples, seed):
    """CEM retains the legal starting chain and records every evaluated chain."""
    rng = np.random.default_rng(seed)
    prefix = np.asarray(prefix).reshape(-1, 2)
    loc = np.asarray(initial_tail).copy()
    scale = np.full_like(loc, 1 / 3)
    actions, costs = [], []
    goal = np.asarray(task["goal"])
    for start in range(0, samples, 12):
        count = min(12, samples - start)
        tail = np.clip(loc + rng.normal(size=(count, *loc.shape)) * scale, -1, 1)
        tail[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (count, *prefix.shape)), tail), axis=1)
        values = np.linalg.norm(exact.endpoints(task, chains) - goal, axis=1)
        actions.extend(chains.tolist())
        costs.extend(values.tolist())
        if count > 1:
            elite = tail[np.argsort(values, kind="stable")[:min(4, count)]]
            loc = elite.mean(axis=0)
            scale = np.maximum(elite.std(axis=0), 1e-5)
    best = int(np.argmin(costs))
    return {"actions": actions, "costs": costs, "best": best}


def heldout_predictions(a, b, costs, parent):
    a_flat = a.reshape(len(a), -1)
    b_flat = b.reshape(len(b), -1)
    nearest = np.argsort(((b_flat[:, None] - a_flat[None]) ** 2).sum(-1), axis=1, kind="stable")[:, :3]
    mean = costs.mean(axis=1)
    relative = costs - mean[:, None]
    return {
        "parent": np.broadcast_to(parent[:, None], (len(parent), len(b))),
        "row_mean": np.broadcast_to(mean[:, None], (len(parent), len(b))),
        "additive": mean[:, None] + relative.mean(axis=0)[nearest].mean(axis=1)[None, :],
        "descriptor": mean[:, None] + relative[:, nearest].mean(axis=2),
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (Path(__file__), PROTOCOL, Path(panel.__file__), Path(exact.__file__), Path(exact.wall.__file__),
             Path(exact.wall.physics.__file__))
    dependencies = {str(path): sha(path) for path in paths}
    tasks = panel.tasks()
    if len(tasks) != 24:
        raise ValueError("panel drift")
    print("Exact wall development: test action-identity diversity against shared lookahead, not visual PC gates.", flush=True)
    rows = []
    for root, task in enumerate(tasks):
        goal = np.asarray(task["goal"])
        nominal_search = optimize(task, [], np.zeros((15, 2)), 120, 261106000 + root)
        candidates = np.asarray(nominal_search["actions"])
        costs = np.asarray(nominal_search["costs"])
        chosen = []
        for index in np.argsort(costs, kind="stable"):
            if all(not np.array_equal(candidates[index, :5], candidates[j, :5]) for j in chosen):
                chosen.append(int(index))
            if len(chosen) == 4:
                break
        if len(chosen) != 4:
            raise ValueError("not enough distinct prefixes")
        global_chains = np.random.default_rng(261107000 + root).uniform(-1, 1, (4, 15, 2))
        pool = np.concatenate((candidates[chosen], global_chains))
        parent = np.linalg.norm(exact.endpoints(task, pool) - goal, axis=1)
        immediate = np.linalg.norm(exact.endpoints(task, pool[:, :5]) - goal, axis=1)
        a = np.random.default_rng(261108000 + root).uniform(-1, 1, (8, 10, 2))
        b = np.random.default_rng(261109000 + root).uniform(-1, 1, (16, 10, 2))

        def cross(tails):
            chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, len(tails), 5, 2)),
                                     np.broadcast_to(tails[None], (8, len(tails), 10, 2))), axis=2)
            return np.linalg.norm(exact.endpoints(task, chains.reshape(-1, 15, 2)) - goal, axis=1).reshape(8, -1)

        a_cost, b_cost = cross(a), cross(b)
        selected = select(a_cost, parent, immediate, 261110000 + root)
        # The evaluator computes all prefix refinements; each arm accesses only its two preselected entries.
        refined = [optimize(task, chain[:5], chain[5:], 36, 261111000 + root * 8 + i)
                   for i, chain in enumerate(pool)]
        outcomes = {}
        for arm, indices in selected.items():
            best_prefix = min(indices, key=lambda i: refined[i]["costs"][refined[i]["best"]])
            result = refined[best_prefix]
            outcomes[arm] = {"prefix_index": best_prefix, "chain": result["actions"][result["best"]],
                             "score": result["costs"][result["best"]]}
        predictions = heldout_predictions(a, b, a_cost, parent)
        mse = {name: float(np.square(values - b_cost).mean()) for name, values in predictions.items()}
        rows.append({"root": root, "task": task, "nominal_search": nominal_search, "nominal_indices": chosen,
                     "pool": pool.tolist(), "parent_cost": parent.tolist(), "immediate": immediate.tolist(),
                     "a": a.tolist(), "a_cost": a_cost.tolist(), "b": b.tolist(), "b_cost": b_cost.tolist(),
                     "heldout_mse": mse, "selected": selected, "refined": refined, "outcomes": outcomes,
                     "deployment_tokens": 120 * 3 + 4 * 3 + 8 + 8 * 8 * 3 + 2 * 36 * 3})
        print(f"root {root}: fixed candidates, A/B and per-prefix refinements complete", flush=True)
    if dependencies != {str(path): sha(path) for path in paths}:
        raise ValueError("dependency changed")
    with OUTPUT.open("x") as stream:
        json.dump({"protocol_sha256": sha(PROTOCOL), "dependencies": dependencies,
                   "scope": "exposed exact-dynamics development, not independent confirmation",
                   "rows": rows}, stream, separators=(",", ":"), allow_nan=False)
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
