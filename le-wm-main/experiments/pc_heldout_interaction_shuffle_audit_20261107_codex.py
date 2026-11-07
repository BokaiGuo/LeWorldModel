"""Post-hoc audit of interaction-specific action-identity permutation controls.

All target costs and outcomes are evaluation only. No model or planner is run.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "results/pc_tworoom_block_heldout_effect_20261107_codex"
POOL = Path(str(STEM) + "_pool.npz")
PLAN = Path(str(STEM) + "_plan.json")
PHYSICAL = Path(str(STEM) + "_physical.json")
OUTPUT = ROOT / "results/pc_heldout_interaction_shuffle_audit_20261107_codex.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(cost):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != (8, 8) or not np.isfinite(cost).all():
        raise ValueError("expected a finite 8x8 matrix")
    mean = cost.mean()
    row = cost.mean(axis=1, keepdims=True)
    column = cost.mean(axis=0, keepdims=True)
    additive = row + column - mean
    return additive, cost - additive


def predict(a, b, cost, permutation_seed=None):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (8, 3, 2) or b.shape != (8, 3, 2):
        raise ValueError("unexpected action shape")
    additive, interaction = decompose(cost)
    if permutation_seed is not None:
        rng = np.random.default_rng(permutation_seed)
        interaction = np.stack([row[rng.permutation(8)] for row in interaction])
        # Re-centering restores both original margins, while retaining a
        # randomized interaction; it does not preserve its exact distribution.
        _, interaction = decompose(interaction)
    nearest = np.argsort(((b[:, None] - a[None]) ** 2).sum(axis=(2, 3)),
                         axis=1, kind="stable")[:, :3]
    return (additive[:, nearest] + interaction[:, nearest]).mean(axis=2)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(PLAN.read_text())
    physical = json.loads(PHYSICAL.read_text())
    if plan["pool_sha256"] != digest(POOL):
        raise ValueError("plan pool hash mismatch")
    if physical["pool_sha256"] != digest(POOL) or physical["plan_sha256"] != digest(PLAN):
        raise ValueError("physical binding mismatch")
    rows = []
    with np.load(POOL, allow_pickle=False) as pool:
        for task, actual in zip(plan["rows"], physical["rows"], strict=True):
            root = task["root"]
            if root != actual["root"] or len(actual["outcomes"]) != 64:
                raise ValueError("physical rows mismatch")
            a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
            cost, target = pool[f"r{root}_a_cost"], pool[f"r{root}_b_cost"]
            identity = predict(a, b, cost)
            if not np.allclose(identity, pool[f"r{root}_pred_identity"], rtol=1e-12, atol=1e-10):
                raise ValueError("original identity prediction mismatch")

            def evaluate(p):
                index = int(np.argmin(p))
                outcome = actual["outcomes"][index]
                return {"mse": float(np.mean((p - target) ** 2)),
                        "index": index, "success": bool(outcome["ever"]),
                        "prefix_success": bool(outcome["prefix_success"]),
                        "final_distance": outcome["final_distance"]}

            shuffled = [evaluate(predict(a, b, cost, 810000 + 1000 * root + k))
                        for k in range(100)]
            rows.append({"root": root, "identity": evaluate(identity),
                         "interaction_shuffle": shuffled})
    summary = {
        "roots": len(rows),
        "identity_success": sum(r["identity"]["success"] for r in rows),
        "identity_post_prefix_success": sum(r["identity"]["success"] and
                                            not r["identity"]["prefix_success"] for r in rows),
        "mean_shuffle_success": float(np.mean([
            sum(r["interaction_shuffle"][k]["success"] for r in rows) for k in range(100)])),
        "mean_identity_mse": float(np.mean([r["identity"]["mse"] for r in rows])),
        "mean_shuffle_mse": float(np.mean([x["mse"] for r in rows for x in r["interaction_shuffle"]])),
        "identity_lower_mse_than_mean_shuffle_roots": int(sum(
            r["identity"]["mse"] < np.mean([x["mse"] for x in r["interaction_shuffle"]]) for r in rows)),
        "mean_identity_distance": float(np.mean([r["identity"]["final_distance"] for r in rows])),
        "mean_shuffle_distance": float(np.mean([x["final_distance"] for r in rows
                                                 for x in r["interaction_shuffle"]])),
    }
    result = {"scope": "post-hoc exposed development diagnostic, not a prospective test",
              "permutations_per_root": 100, "summary": summary, "rows": rows,
              "pool_sha256": digest(POOL), "plan_sha256": digest(PLAN),
              "physical_sha256": digest(PHYSICAL), "runner_sha256": digest(Path(__file__))}
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
