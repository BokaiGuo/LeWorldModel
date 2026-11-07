"""Post-selection full-pool coverage audit; never modifies sealed decisions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as base
from experiments import pc_exact_descriptor_combination_proposal_20261107_codex as src
from experiments import pc_exact_goal_relative_bridge_20261107_codex as geom

PROTOCOL = geom.ROOT.parent / "docs/PC_EXACT_COMBINATION_FULL_POOL_COVERAGE_PROTOCOL_20261107_CODEX.md"
OUTPUT = geom.ROOT / "results/pc_exact_combination_full_pool_coverage_20261107_codex.json"
TOL = 1e-10


def summarize(parent, a_cost, b_cost, pairs_by_arm):
    parent, a_cost, b_cost = map(lambda x: np.asarray(x, dtype=float), (parent, a_cost, b_cost))
    if parent.shape != (8,) or a_cost.shape != (8, 8) or b_cost.shape != (8, 64):
        raise ValueError("invalid pool dimensions")
    if not all(np.isfinite(x).all() for x in (parent, a_cost, b_cost)):
        raise ValueError("nonfinite costs")
    incumbent = float(min(parent.min(), a_cost.min()))
    best = float(min(incumbent, b_cost.min()))
    better = b_cost < incumbent - TOL
    success = b_cost < .05
    new_success = bool(incumbent >= .05 and success.any())
    arms = {}
    for arm, pairs in pairs_by_arm.items():
        indices = np.asarray(pairs)
        if indices.shape != (8,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("expected eight integer indices")
        if len(np.unique(indices)) != 8 or np.any(indices < 0) or np.any(indices >= 512):
            raise ValueError("duplicate or invalid pair")
        queried = b_cost.ravel()[indices]
        chosen = min(incumbent, float(queried.min()))
        arms[arm] = {"queried_improving": int(better.ravel()[indices].sum()),
                     "queried_successes": int(success.ravel()[indices].sum()),
                     "found_new_success": bool(incumbent >= .05 and (queried < .05).any()),
                     "selected_score": chosen, "regret_to_pool_best": chosen - best}
    return {"incumbent": incumbent, "pool_best": best, "improving_combinations": int(better.sum()),
            "success_combinations": int(success.sum()), "pool_has_new_success": new_success, "arms": arms}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    historical = json.loads(src.OUTPUT.read_text())
    for path, expected in historical["dependencies"].items():
        if base.sha(Path(path)) != expected:
            raise ValueError(f"historical dependency drift: {path}")
    if len(historical["rows"]) != 24:
        raise ValueError("unexpected root count")
    rows = []
    for root, (row, task) in enumerate(zip(historical["rows"], src.tasks(), strict=True)):
        if row["root"] != root or row["task"] != task:
            raise ValueError("root/task mismatch")
        offsets = np.asarray(row["offsets"])
        if offsets.shape != (80,):
            raise ValueError("invalid offsets")
        chains = np.stack([geom.make_chain(task["start"], task["goal"], x)[0] for x in offsets])
        pool, a, b = chains[:8], chains[8:16, 5:], chains[16:, 5:]
        a_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 8, 5, 2)),
                                   np.broadcast_to(a[None], (8, 8, 10, 2))), axis=2)
        b_chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, 64, 5, 2)),
                                   np.broadcast_to(b[None], (8, 64, 10, 2))), axis=2)
        def score(actions):
            return np.linalg.norm(base.exact.endpoints(task, actions) - np.asarray(task["goal"]), axis=1)
        parent = score(pool)
        a_cost = score(a_chains.reshape(-1, 15, 2)).reshape(8, 8)
        np.testing.assert_allclose(parent, row["parent_cost"], rtol=0, atol=TOL)
        np.testing.assert_allclose(a_cost, row["a_cost"], rtol=0, atol=TOL)
        b_cost = score(b_chains.reshape(-1, 15, 2)).reshape(8, 64)
        outcomes = row["outcomes"]
        common_chains = np.concatenate((pool, a_chains.reshape(-1, 15, 2)))
        common_costs = np.concatenate((parent, a_cost.ravel()))
        common_best = int(np.argmin(common_costs))
        for arm, outcome in outcomes.items():
            ids = np.asarray(outcome["pairs"])
            np.testing.assert_allclose(b_cost.ravel()[ids], outcome["queried_costs"], rtol=0, atol=TOL)
            np.testing.assert_allclose(score(np.asarray(outcome["chain"])[None]), [outcome["score"]], rtol=0, atol=TOL)
            best_query = int(np.argmin(b_cost.ravel()[ids]))
            from_b = b_cost.ravel()[ids[best_query]] < common_costs[common_best]
            expected = b_chains.reshape(-1, 15, 2)[ids[best_query]] if from_b else common_chains[common_best]
            if outcome["origin"] != ("queried_B" if from_b else "common"):
                raise ValueError(f"origin mismatch: {arm}")
            np.testing.assert_allclose(outcome["chain"], expected, rtol=0, atol=TOL)
        result = summarize(parent, a_cost, b_cost, {arm: x["pairs"] for arm, x in outcomes.items()})
        for arm, outcome in outcomes.items():
            np.testing.assert_allclose(result["arms"][arm]["selected_score"], outcome["score"], rtol=0, atol=TOL)
        rows.append({"root": root, "task": task, "b_costs": b_cost.tolist(), **result})
    summary = {"roots": len(rows), "pool_improves_roots": sum(r["improving_combinations"] > 0 for r in rows),
               "pool_new_success_roots": sum(r["pool_has_new_success"] for r in rows),
               "arms": {arm: {"finds_improvement_roots": sum(r["arms"][arm]["queried_improving"] > 0 for r in rows),
                              "finds_new_success_roots": sum(r["arms"][arm]["found_new_success"] for r in rows),
                              "mean_regret": float(np.mean([r["arms"][arm]["regret_to_pool_best"] for r in rows]))}
                        for arm in src.ARMS}}
    output = {"protocol": base.sha(PROTOCOL), "runner": base.sha(Path(__file__)),
              "historical": base.sha(src.OUTPUT), "roots": rows, "summary": summary,
              "evaluation_only_b_queries": 24 * 512,
              "scope": "post-hoc exact-dynamics pool ceiling; no change to deployed actions or gates"}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
