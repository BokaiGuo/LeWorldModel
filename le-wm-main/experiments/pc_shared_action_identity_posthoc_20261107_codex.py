"""Post-hoc shared-action interaction controls; no new model or environment queries."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "results/pc_tworoom_block_heldout_effect_20261107_codex"
OUTPUT = ROOT / "results/pc_shared_action_identity_posthoc_20261107_codex.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_actions(chains, actions):
    if chains.shape != (8, 8, 25, 2) or actions.shape != (8, 3, 2):
        raise ValueError("unexpected action shapes")
    expected = np.broadcast_to(np.repeat(actions, 5, axis=1)[None], (8, 8, 15, 2))
    if not np.array_equal(chains[:, :, 10:], expected):
        raise ValueError("tails not shared across prefixes")
    prefixes = np.broadcast_to(chains[:, :1, :10], (8, 8, 10, 2))
    if not np.array_equal(chains[:, :, :10], prefixes):
        raise ValueError("prefix varies across tails")
    if not np.isfinite(chains).all() or np.abs(chains).max() > 1:
        raise ValueError("invalid action")


def predict(a, b, costs):
    """Additive fit and conditional residuals mapped by three nearest A actions."""
    if a.shape != (8, 3, 2) or b.shape != a.shape or costs.shape != (8, 8):
        raise ValueError("unexpected input shapes")
    if not all(np.isfinite(x).all() for x in (a, b, costs)):
        raise ValueError("nonfinite input")
    nearest = np.argsort(((b[:, None] - a[None]) ** 2).sum(axis=(2, 3)),
                         axis=1, kind="stable")[:, :3]
    row = costs.mean(axis=1, keepdims=True)
    col = costs.mean(axis=0, keepdims=True)
    grand = costs.mean()
    residual = costs - row - col + grand
    additive = row + (col - grand)[0, nearest].mean(axis=1)[None, :]

    def assemble(interaction):
        return additive + interaction[:, nearest].mean(axis=2)

    matched = assemble(residual)
    row_shuffles = np.stack([assemble(np.roll(residual, shift, axis=0)) for shift in range(1, 8)])
    col_shuffles = np.stack([assemble(np.roll(residual, shift, axis=1)) for shift in range(1, 8)])
    return additive, matched, row_shuffles, col_shuffles


def measure(pred, truth, outcomes):
    i = int(np.argmin(pred))
    item = outcomes[i]
    return {"mse": float(np.mean((pred - truth) ** 2)),
            "choice": i, "ever": bool(item["ever"]),
            "tail_success": bool(item["ever"] and not item["prefix_success"]),
            "final_distance": float(item["final_distance"])}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    pool_path = Path(str(STEM) + "_pool.npz")
    plan_path = Path(str(STEM) + "_plan.json")
    physical_path = Path(str(STEM) + "_physical.json")
    plan = json.loads(plan_path.read_text())
    physical = json.loads(physical_path.read_text())
    if plan["pool_sha256"] != sha(pool_path):
        raise ValueError("pool binding mismatch")
    if physical["plan_sha256"] != sha(plan_path) or physical["pool_sha256"] != sha(pool_path):
        raise ValueError("physical binding mismatch")
    rows = []
    with np.load(pool_path, allow_pickle=False) as pool:
        for p, q in zip(plan["rows"], physical["rows"], strict=True):
            root = p["root"]
            if root != q["root"] or p["row"] != q["row"] or len(q["outcomes"]) != 64:
                raise ValueError("record mismatch")
            a, b = pool[f"r{root}_a"], pool[f"r{root}_b"]
            for label, actions in (("a", a), ("b", b)):
                validate_actions(pool[f"r{root}_{label}_actions"], actions)
            ca, cb = pool[f"r{root}_a_cost"], pool[f"r{root}_b_cost"]
            additive, matched, row_shuffle, col_shuffle = predict(a, b, ca)
            if not np.allclose(matched, pool[f"r{root}_pred_identity"], rtol=1e-12, atol=1e-9):
                raise ValueError("matched method differs from prior identity implementation")
            outcomes = q["outcomes"]
            rows.append({"root": root, "row": p["row"],
                         "additive": measure(additive, cb, outcomes),
                         "matched": measure(matched, cb, outcomes),
                         "row_shuffle": [measure(x, cb, outcomes) for x in row_shuffle],
                         "column_shuffle": [measure(x, cb, outcomes) for x in col_shuffle]})
    if len(rows) != 10 or len({r["root"] for r in rows}) != 10:
        raise ValueError("incomplete roots")
    result = {"scope": "posthoc exposed roots; no new model calls; not prospective or independent",
              "definition": "double-centered A interaction is cyclically reassigned across prefixes or shared A actions; additive components remain fixed",
              "shuffle_shifts": list(range(1, 8)), "rows": rows,
              "bindings": {str(p): sha(p) for p in (pool_path, plan_path, physical_path, Path(__file__))}}
    with OUTPUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
