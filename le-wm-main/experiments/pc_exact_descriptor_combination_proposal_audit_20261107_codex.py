"""Audit candidate identities, prediction selection and actually queried chains."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_descriptor_combination_proposal_20261107_codex as run

OUTPUT = run.geom.ROOT / "results/pc_exact_descriptor_combination_proposal_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    if result["dependencies"] != {p: run.base.sha(p) for p in result["dependencies"]}:
        raise ValueError("dependency drift")
    rows = []
    for root, (entry, task) in enumerate(zip(result["rows"], run.tasks(), strict=True)):
        if entry["root"] != root or entry["task"] != task:
            raise ValueError("task mismatch")
        offsets = np.random.default_rng(2720000 + root).uniform(-2, 2, 80)
        np.testing.assert_array_equal(offsets, entry["offsets"])
        chains = np.stack([run.geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        goal = np.asarray(task["goal"])

        def score(x):
            return np.linalg.norm(run.base.exact.endpoints(task, x) - goal, axis=1)

        parents = score(pool)
        np.testing.assert_allclose(parents, entry["parent_cost"], atol=1e-12, rtol=0)
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2).reshape(-1, 15, 2)
        ac = score(a_chains).reshape(8, 8)
        np.testing.assert_allclose(ac, entry["a_cost"], atol=1e-12, rtol=0)
        pred = run.proposal_predictions(a, b, ac, parents, 2721000 + root)
        common = np.r_[parents, ac.ravel()]
        common_chains = np.concatenate((pool, a_chains))
        for arm, outcome in entry["outcomes"].items():
            expected = (np.random.default_rng(2722000 + root).choice(512, 8, replace=False).tolist()
                        if arm == "random" else run.choose(pred[arm]))
            if outcome["pairs"] != expected or len(set(expected)) != 8:
                raise ValueError("candidate selection mismatch")
            pair = np.asarray(expected)
            queried = np.concatenate((pool[pair // 64, :5], b[pair % 64]), axis=1)
            values = score(queried)
            np.testing.assert_allclose(values, outcome["queried_costs"], atol=1e-12, rtol=0)
            best_common, best_query = int(np.argmin(common)), int(np.argmin(values))
            if values[best_query] < common[best_common]:
                best, index, origin = queried, best_query, "queried_B"
                value = values[best_query]
            else:
                best, index, origin = common_chains, best_common, "common"
                value = common[best_common]
            if origin != outcome["origin"] or abs(value - outcome["score"]) > 1e-12:
                raise ValueError("outcome mismatch")
            np.testing.assert_array_equal(best[index], outcome["chain"])
        if entry["per_arm_dynamics_steps"] != 1200:
            raise ValueError("budget mismatch")
        rows.append({"root": root, "arms": len(entry["outcomes"]), "queried_per_arm": 8})
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "source_sha256": run.base.sha(run.OUTPUT),
                   "runner_sha256": run.base.sha(Path(__file__)),
                   "scope": "same exact dynamics and prediction helpers, not independent engine"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} tasks, 5 arms and 8 queried chains per arm")


if __name__ == "__main__":
    audit()
