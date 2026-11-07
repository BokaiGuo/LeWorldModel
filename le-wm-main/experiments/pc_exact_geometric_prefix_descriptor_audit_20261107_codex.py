"""Recompute every stored refinement and fixed-pool decision using the same exact dynamics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_geometric_prefix_descriptor_20261107_codex as run

OUTPUT = run.geom.ROOT / "results/pc_exact_geometric_prefix_descriptor_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    source = json.loads(run.geom.OUTPUT.read_text())
    if result["dependencies"] != {path: run.base.sha(path) for path in result["dependencies"]}:
        raise ValueError("source drift")
    rows = []
    for root, (entry, original) in enumerate(zip(result["rows"], source["rows"], strict=True)):
        if entry["root"] != root or original["root"] != root or entry["task"] != original["task"]:
            raise ValueError("root mismatch")
        ids = np.random.default_rng(2717000 + root).choice(384, size=24, replace=False)
        np.testing.assert_array_equal(ids, entry["source_indices"])
        actions = np.asarray(original["actions"])
        pool = actions[ids[:8]]
        a, b = actions[ids[8:16], 5:], actions[ids[16:], 5:]
        goal = np.asarray(entry["task"]["goal"])

        def cost(chains):
            return np.linalg.norm(run.base.exact.endpoints(entry["task"], chains) - goal, axis=1)

        def cross(tails):
            chains = np.concatenate((np.broadcast_to(pool[:, None, :5], (8, len(tails), 5, 2)),
                                     np.broadcast_to(tails[None], (8, len(tails), 10, 2))), axis=2)
            return cost(chains.reshape(-1, 15, 2)).reshape(8, -1)

        np.testing.assert_allclose(cost(pool), entry["parent_cost"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(cost(pool[:, :5]), entry["immediate"], atol=1e-10, rtol=0)
        a_cost, b_cost = cross(a), cross(b)
        hold = cross(np.zeros((1, 10, 2)))[:, 0]
        np.testing.assert_allclose(a_cost, entry["a_cost"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(b_cost, entry["b_cost"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(hold[:, None] - a_cost, entry["effect"], atol=1e-10, rtol=0)
        selected = run.base.select(a_cost, entry["parent_cost"], entry["immediate"], 2718000 + root)
        if selected != entry["selected"]:
            raise ValueError("selection mismatch")
        max_cost_error = 0.
        for i, ref in enumerate(entry["refined"]):
            chains = np.asarray(ref["actions"])
            if chains.shape != (36, 15, 2) or np.max(np.abs(chains)) > 1 + 1e-12:
                raise ValueError("illegal refinement")
            np.testing.assert_allclose(chains[:, :5], np.broadcast_to(pool[i, :5], (36, 5, 2)), atol=0, rtol=0)
            calculated = cost(chains)
            max_cost_error = max(max_cost_error, float(np.max(np.abs(calculated - np.asarray(ref["costs"])))))
            if int(np.argmin(calculated)) != ref["best"]:
                raise ValueError("best refinement mismatch")
            reconstructed = run.base.optimize(entry["task"], pool[i, :5], pool[i, 5:], 36,
                                               2719000 + 8 * root + i)
            np.testing.assert_allclose(reconstructed["actions"], chains, atol=1e-12, rtol=0)
        if max_cost_error > 1e-10:
            raise ValueError("refinement score mismatch")
        for arm in run.ARMS:
            outcome = run.select_with_incumbent(pool, entry["parent_cost"], entry["refined"], selected[arm])
            if outcome != entry["outcomes"][arm]:
                raise ValueError("outcome mismatch")
        preds = run.base.heldout_predictions(a, b, a_cost, np.asarray(entry["parent_cost"]))
        for name, values in preds.items():
            np.testing.assert_allclose(np.square(values - b_cost).mean(), entry["heldout_mse"][name], atol=1e-12)
        rows.append({"root": root, "max_refinement_cost_error": max_cost_error,
                     "refinements_checked": 8 * 36, "deploy_dynamics_steps": entry["deployment_dynamics_steps"]})
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "source_sha256": run.base.sha(run.OUTPUT),
                   "runner_sha256": run.base.sha(Path(__file__)),
                   "scope": "full replay through same dynamics backend, not independent physics"},
                  stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} roots and {len(rows) * 8 * 36} refinement chains")


if __name__ == "__main__":
    audit()
