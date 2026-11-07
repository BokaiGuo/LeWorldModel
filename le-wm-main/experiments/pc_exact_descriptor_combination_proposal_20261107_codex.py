"""Predict then evaluate unqueried exact-dynamics prefix-tail combinations."""
import itertools
import json
import time
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as base
from experiments import pc_exact_goal_relative_bridge_20261107_codex as geom

PROTOCOL = geom.ROOT.parent / "docs/PC_EXACT_DESCRIPTOR_COMBINATION_PROPOSAL_PROTOCOL_20261107_CODEX.md"
OUTPUT = geom.ROOT / "results/pc_exact_descriptor_combination_proposal_20261107_codex.json"
ARMS = ("row_mean", "additive", "descriptor", "shuffled", "random")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.375, -.075, .075, .375),
                                                   (-np.pi / 6, 0., np.pi / 6), (-.25, .25))]


def choose(prediction, k=8):
    values = np.asarray(prediction, dtype=float)
    if values.shape != (8, 64) or not np.isfinite(values).all() or not 0 < k <= values.size:
        raise ValueError("expected finite 8x64 prediction")
    return np.argsort(values.ravel(), kind="stable")[:k].tolist()


def proposal_predictions(a, b, a_cost, parent, seed):
    predictions = base.heldout_predictions(a, b, a_cost, parent)
    rng = np.random.default_rng(seed)
    shuffled = np.stack([row[rng.permutation(len(row))] for row in a_cost])
    predictions["shuffled"] = base.heldout_predictions(a, b, shuffled, parent)["descriptor"]
    return predictions


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    dependencies = {str(p): base.sha(p) for p in (
        Path(__file__), PROTOCOL, Path(geom.__file__), Path(base.__file__),
        Path(base.exact.__file__), Path(base.exact.wall.__file__), Path(base.exact.wall.physics.__file__))}
    rows = []
    print("Exact development: predict unqueried legal combinations before eight actual queries per arm.", flush=True)
    for root, task in enumerate(tasks()):
        started = time.perf_counter()
        offsets = np.random.default_rng(2720000 + root).uniform(-2, 2, 80)
        chains = np.stack([geom.make_chain(task["start"], task["goal"], o)[0] for o in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        if (chains.shape != (80, 15, 2) or not np.isfinite(chains).all()
                or np.max(np.abs(chains)) > 1 + 1e-12):
            raise ValueError("illegal candidate action")
        goal = np.asarray(task["goal"])

        def score(x):
            return np.linalg.norm(base.exact.endpoints(task, x) - goal, axis=1)

        parent_cost = score(pool)
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2)
        a_cost = score(a_chains.reshape(-1, 15, 2)).reshape(8, 8)
        predictions = proposal_predictions(a, b, a_cost, parent_cost, 2721000 + root)
        selections = {arm: choose(predictions[arm]) for arm in ARMS if arm != "random"}
        selections["random"] = np.random.default_rng(2722000 + root).choice(512, 8, replace=False).tolist()
        common_cost = np.r_[parent_cost, a_cost.ravel()]
        common_index = int(np.argmin(common_cost))
        common_chains = np.concatenate((pool, a_chains.reshape(-1, 15, 2)))
        outcomes = {}
        for arm in ARMS:
            pairs = np.asarray(selections[arm])
            proposal_chains = np.concatenate((pool[pairs // 64, :5], b[pairs % 64]), axis=1)
            actual = score(proposal_chains)
            best = int(np.argmin(actual))
            if actual[best] < common_cost[common_index]:
                chain, value, origin = proposal_chains[best], float(actual[best]), "queried_B"
            else:
                chain, value, origin = common_chains[common_index], float(common_cost[common_index]), "common"
            outcomes[arm] = {"pairs": pairs.tolist(), "queried_costs": actual.tolist(), "origin": origin,
                             "score": value, "chain": chain.tolist()}
        rows.append({"root": root, "task": task, "offsets": offsets.tolist(), "parent_cost": parent_cost.tolist(),
                     "a_cost": a_cost.tolist(), "outcomes": outcomes, "per_arm_dynamics_steps": 80 * 15,
                     "elapsed_all_arms_seconds": time.perf_counter() - started})
        print(f"root {root}: additive={outcomes['additive']['score']:.5f}, "
              f"descriptor={outcomes['descriptor']['score']:.5f}", flush=True)
    if dependencies != {path: base.sha(path) for path in dependencies}:
        raise ValueError("dependency drift")
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exact-state new configurations in exposed task family, not independent confirmation"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    run()
