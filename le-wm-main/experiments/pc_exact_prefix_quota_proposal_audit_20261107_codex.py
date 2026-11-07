"""Reconstruct selections and actual actions; uses same dynamics backend."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_prefix_quota_proposal_20261107_codex as run

OUTPUT = run.geom.ROOT / "results/pc_exact_prefix_quota_proposal_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(run.OUTPUT.read_text())
    if data["dependencies"] != {p: run.base.sha(p) for p in data["dependencies"]}:
        raise ValueError("dependency drift")
    rows = []
    for root, (row, task) in enumerate(zip(data["rows"], run.tasks(), strict=True)):
        if row["root"] != root or row["task"] != task or row["per_arm_model_steps"] != 1200:
            raise ValueError("task or budget mismatch")
        offsets = np.random.default_rng(2730000 + root).uniform(-2, 2, 80)
        np.testing.assert_array_equal(offsets, row["offsets"])
        chains = np.stack([run.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        def score(x):
            return np.linalg.norm(run.base.exact.endpoints(task, x) - np.asarray(task["goal"]), axis=1)
        parent = score(pool)
        a_chain = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                  np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2).reshape(64, 15, 2)
        ac = score(a_chain).reshape(8, 8)
        np.testing.assert_allclose(parent, row["parent_cost"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(ac, row["a_cost"], atol=1e-12, rtol=0)
        predictions = run.prior.proposal_predictions(a, b, ac, parent, 2731000 + root)
        common = np.r_[parent, ac.ravel()]
        common_actions = np.concatenate((pool, a_chain))
        expected = {
            "global_descriptor": np.argsort(predictions["descriptor"].ravel(), kind="stable")[:8].tolist(),
            "quota_descriptor": [i * 64 + int(np.argmin(predictions["descriptor"][i])) for i in range(8)],
            "quota_additive": [i * 64 + int(np.argmin(predictions["additive"][i])) for i in range(8)],
            "quota_shuffled": [i * 64 + int(np.argmin(predictions["shuffled"][i])) for i in range(8)],
            "quota_random": (np.arange(8) * 64 + np.random.default_rng(2732000 + root).integers(0, 64, size=8)).tolist(),
        }
        for arm, ids in expected.items():
            outcome = row["outcomes"][arm]
            if outcome["pairs"] != ids or len(set(ids)) != 8:
                raise ValueError(f"selection mismatch {root} {arm}")
            ix = np.asarray(ids)
            candidates = np.concatenate((pool[ix // 64, :5], b[ix % 64]), axis=1)
            costs = score(candidates)
            np.testing.assert_allclose(costs, outcome["queried_costs"], atol=1e-12, rtol=0)
            q, c = int(np.argmin(costs)), int(np.argmin(common))
            use_q = costs[q] < common[c]
            np.testing.assert_array_equal(outcome["chain"], candidates[q] if use_q else common_actions[c])
            np.testing.assert_allclose(outcome["score"], costs[q] if use_q else common[c], atol=1e-12, rtol=0)
            if outcome["origin"] != ("queried_B" if use_q else "common"):
                raise ValueError("origin mismatch")
        rows.append({"root": root, "actual_queried_per_arm": 8,
                     "different_quota_vs_global_pairs": len(set(expected["quota_descriptor"]) -
                                                            set(expected["global_descriptor"]))})
    with OUTPUT.open("x") as stream:
        json.dump({"source": run.base.sha(run.OUTPUT), "runner": run.base.sha(Path(__file__)),
                   "rows": rows, "scope": "same prediction helpers and exact dynamics; not independent engine"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} roots; changed pairs {sum(r['different_quota_vs_global_pairs'] for r in rows)}")


if __name__ == "__main__":
    audit()
