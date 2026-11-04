"""Post-seal audit of the actual selected chains, including gradient selections."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_exact_unicycle_development as base

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_exact_orientation_mechanism_20261103_codex.json"
OUTPUT = ROOT / "results/pc_exact_orientation_selected_audit_20261103_codex.json"


def audit(row):
    goal = np.asarray(row["goal"])
    chain = np.asarray(row["combined_actions"])
    prefix, suffix = chain
    successor = base.step(np.zeros(3), prefix)
    initial = np.linalg.norm(goal)
    prefix_cost = np.linalg.norm(successor[:2] - goal)
    selected_cost, _ = base.evaluate(chain, goal)
    zero_same, _ = base.evaluate(np.stack([np.zeros(2), suffix]), goal)
    zero_best, _ = base.evaluate(np.array([[0., 0.], [np.clip(goal[0], -1., 1.), 0.]]), goal)
    response = np.array([np.cos(successor[2]), np.sin(successor[2])])
    velocity = np.clip(np.dot(goal - successor[:2], response), -1., 1.)
    selected_best, _ = base.evaluate(np.stack([prefix, [velocity, 0.]]), goal)
    selected_active = selected_cost - prefix_cost
    zero_active = zero_same - initial
    nonprogress = prefix_cost >= initial - 1e-12
    witness = (nonprogress and selected_active < zero_active - 1e-12
               and selected_best < zero_best - 1e-12)
    return {"index": row["index"], "source": row["combined_selected"],
            "initial_distance": float(initial), "prefix_distance": float(prefix_cost),
            "selected_chain_distance": selected_cost, "zero_same_suffix_distance": zero_same,
            "selected_active_change": float(selected_active), "zero_active_change": float(zero_active),
            "selected_prefix_reoptimized_distance": selected_best,
            "zero_prefix_reoptimized_distance": zero_best,
            "nonprogress": bool(nonprogress), "witness": bool(witness)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text())
    rows = [audit(row) for row in source["rows"]]
    report = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "post-seal exact-dynamics audit, not a new planner selection",
              "nonprogress_count": sum(x["nonprogress"] for x in rows),
              "witness_count": sum(x["witness"] for x in rows), "rows": rows}
    with OUTPUT.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("nonprogress_count", "witness_count")}))


if __name__ == "__main__":
    main()
