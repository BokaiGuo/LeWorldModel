"""Evaluate one-query-per-prefix allocation with sealed action-effect predictions."""
import itertools
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as base
from experiments import pc_exact_descriptor_combination_proposal_20261107_codex as prior
from experiments import pc_exact_goal_relative_bridge_20261107_codex as geom

PROTOCOL = geom.ROOT.parent / "docs/PC_EXACT_PREFIX_QUOTA_PROPOSAL_PROTOCOL_20261107_CODEX.md"
OUTPUT = geom.ROOT / "results/pc_exact_prefix_quota_proposal_20261107_codex.json"
ARMS = ("global_descriptor", "quota_descriptor", "quota_additive", "quota_shuffled", "quota_random")


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.32, -.12, .12, .32),
                                                   (-np.pi / 9, 0., np.pi / 9), (-.2, .2))]


def select(predictions, seed):
    desc, additive, shuffled = (np.asarray(predictions[key]) for key in ("descriptor", "additive", "shuffled"))
    if any(x.shape != (8, 64) or not np.isfinite(x).all() for x in (desc, additive, shuffled)):
        raise ValueError("expected finite 8x64 predictions")
    def quota(x):
        return (np.arange(8) * 64 + np.argmin(x, axis=1)).tolist()
    return {"global_descriptor": prior.choose(desc), "quota_descriptor": quota(desc),
            "quota_additive": quota(additive), "quota_shuffled": quota(shuffled),
            "quota_random": (np.arange(8) * 64 + np.random.default_rng(seed).integers(0, 64, size=8)).tolist()}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [Path(__file__), PROTOCOL, Path(prior.__file__), Path(geom.__file__), Path(base.__file__),
             Path(base.exact.__file__), Path(base.exact.wall.__file__), Path(base.exact.wall.physics.__file__)]
    dependencies = {str(p): base.sha(p) for p in paths}
    rows = []
    print("Exact wall development: compare equal-budget global and per-prefix selections.", flush=True)
    for root, task in enumerate(tasks()):
        offsets = np.random.default_rng(2730000 + root).uniform(-2, 2, 80)
        chains = np.stack([geom.make_chain(task["start"], task["goal"], o)[0] for o in offsets])
        if chains.shape != (80, 15, 2) or not np.isfinite(chains).all() or np.abs(chains).max() > 1 + 1e-12:
            raise ValueError("invalid actions")
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        def score(actions):
            return np.linalg.norm(base.exact.endpoints(task, actions) - np.asarray(task["goal"]), axis=1)
        parent = score(pool)
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2)
        a_cost = score(a_chains.reshape(-1, 15, 2)).reshape(8, 8)
        pred = prior.proposal_predictions(a, b, a_cost, parent, 2731000 + root)
        selected = select(pred, 2732000 + root)
        common_chains = np.concatenate((pool, a_chains.reshape(-1, 15, 2)))
        common_costs = np.concatenate((parent, a_cost.ravel()))
        common_best = int(np.argmin(common_costs))
        outcomes = {}
        for arm, pairs in selected.items():
            ids = np.asarray(pairs)
            proposals = np.concatenate((pool[ids // 64, :5], b[ids % 64]), axis=1)
            costs = score(proposals)
            best = int(np.argmin(costs))
            use_b = costs[best] < common_costs[common_best]
            chain = proposals[best] if use_b else common_chains[common_best]
            value = float(costs[best] if use_b else common_costs[common_best])
            outcomes[arm] = {"pairs": pairs, "queried_costs": costs.tolist(), "score": value,
                             "origin": "queried_B" if use_b else "common", "chain": chain.tolist()}
        rows.append({"root": root, "task": task, "offsets": offsets.tolist(), "parent_cost": parent.tolist(),
                     "a_cost": a_cost.tolist(), "outcomes": outcomes, "per_arm_model_steps": 80 * 15})
    if dependencies != {path: base.sha(Path(path)) for path in dependencies}:
        raise ValueError("dependency changed")
    summary = {}
    for arm in ARMS:
        values = [r["outcomes"][arm]["score"] for r in rows]
        summary[arm] = {"successes": sum(x < .05 for x in values), "mean_distance": float(np.mean(values))}
    desc = [r["outcomes"]["quota_descriptor"]["score"] < .05 for r in rows]
    for arm in ("quota_additive", "quota_shuffled", "global_descriptor"):
        other = [r["outcomes"][arm]["score"] < .05 for r in rows]
        summary[arm]["descriptor_only"] = sum(a and not b for a, b in zip(desc, other, strict=True))
        summary[arm]["comparator_only"] = sum(b and not a for a, b in zip(desc, other, strict=True))
    passed = all(summary[arm]["descriptor_only"] >= 2 and summary[arm]["comparator_only"] <= 1
                 for arm in ("quota_additive", "quota_shuffled"))
    passed &= summary["quota_descriptor"]["successes"] >= summary["global_descriptor"]["successes"]
    with OUTPUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows, "summary": summary, "development_exit": bool(passed),
                   "scope": "exact state and dynamics, exposed task family; not RGB-only or independent"},
                  stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"summary": summary, "development_exit": bool(passed)}, indent=2))


if __name__ == "__main__":
    run()
