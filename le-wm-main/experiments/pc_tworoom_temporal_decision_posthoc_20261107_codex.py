"""Post-hoc fixed-pool selection; real outcomes are evaluator-only."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_within_prefix_pairwise_posthoc_20261107_codex as base

OUT = base.ROOT / "results/pc_tworoom_temporal_decision_posthoc_20261107_codex.json"
METHODS = ("step15", "step20", "step25", "minimum_all_steps")


def selected_indices(cost):
    cost = np.asarray(cost)
    if cost.shape != (64, 5) or not np.isfinite(cost).all():
        raise ValueError("expected finite 64 by 5 cost matrix")
    return {"step15": int(np.argmin(cost[:, 2])),
            "step20": int(np.argmin(cost[:, 3])),
            "step25": int(np.argmin(cost[:, 4])),
            "minimum_all_steps": int(np.argmin(cost.min(axis=1)))}


def main():
    plan, physical = base.check_bindings()
    rows = []
    with np.load(base.POOL, allow_pickle=False) as pool, np.load(base.FEATURES, allow_pickle=False) as features:
        for root in range(10):
            record = physical["rows"][root]
            if record["root"] != root or record["row"] != plan["rows"][root]["row"]:
                raise ValueError("root binding mismatch")
            base.validate(pool[f"r{root}_b_actions"], record["outcomes"])
            cost = features[f"r{root}_predicted_cost"]
            choices = {}
            for name, idx in selected_indices(cost).items():
                outcome = record["outcomes"][idx]
                choices[name] = {"index": idx, "ever": outcome["ever"],
                                 "prefix_success": outcome["prefix_success"],
                                 "terminal_success": outcome["trace"][-1]["terminated"],
                                 "final_distance": outcome["final_distance"]}
            rows.append({"root": root, "row": record["row"], "choices": choices,
                         "oracle_pool_ever": any(o["ever"] for o in record["outcomes"])})
    summary = {}
    for method in METHODS:
        comp = [(row["choices"][method]["ever"], row["choices"]["step25"]["ever"]) for row in rows]
        summary[method] = {"ever_success": sum(x for x, _ in comp),
                           "terminal_success": sum(row["choices"][method]["terminal_success"] for row in rows),
                           "exclusive_win_vs_step25": sum(x and not y for x, y in comp),
                           "exclusive_loss_vs_step25": sum(not x and y for x, y in comp),
                           "different_chain_vs_step25": sum(row["choices"][method]["index"] !=
                                                            row["choices"]["step25"]["index"] for row in rows)}
    payload = {"scope": "posthoc development, fixed pool; not an online planner comparison",
               "source_hashes": {str(p): base.digest(p) for p in
                                 (base.PLAN, base.POOL, base.PHYSICAL, base.RESULT, base.FEATURES,
                                  Path(base.__file__), Path(__file__))},
               "rows": rows, "summary": summary}
    with OUT.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
