"""Analyze independent tail optimization and parity-gated real effects."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sign(x):
    return 0 if abs(x) <= 1e-8 else 1 if x > 0 else -1


def main():
    path = (Path(__file__).resolve().parents[1] / "results" /
            "pc_prefix_heldout_optimization_replay_seed42_20261105_codex.json")
    data = json.loads(path.read_text())
    if (data["errors"] or len(data["parity"]) != 50 or
            not all(x["pass"] for x in data["parity"]) or len(data["counterfactual"]) != 50):
        raise ValueError("parity or task census incomplete")
    totals = Counter()
    ranks = {name: Counter() for name in ("parent", "a_min", "a_gain")}
    for task in data["counterfactual"]:
        p = np.asarray(task["parent_cost"])
        a = np.asarray(task["construction_probe_cost"]).min(1)
        opt = np.asarray(task["optimized_cost"])
        for name, values, target in (("parent", p, opt), ("a_min", a, opt),
                                     ("a_gain", p - a, p - opt)):
            x, y = sign(values[0] - values[1]), sign(target[0] - target[1])
            ranks[name]["tie" if 0 in (x, y) else "agree" if x == y else "opposite"] += 1
        optimized = [x for x in task["chains"] if x["chain"] == "optimized"]
        parent = [x for x in task["chains"] if x["chain"] == "parent"]
        if len(optimized) != 2 or len(parent) != 2:
            raise ValueError("incomplete branch outcomes")
        winners = [x["success"] for x in optimized]
        totals["optimized_success_chains"] += sum(winners)
        totals["parent_success_chains"] += sum(x["success"] for x in parent)
        totals["optimized_illegal_plans"] += sum(x["out_of_bounds_plan"] > 0 for x in optimized)
        totals["optimized_illegal_executed"] += sum(x["out_of_bounds_executed"] > 0 for x in optimized)
        key = "both" if all(winners) else "one" if any(winners) else "neither"
        totals[key] += 1
        totals["selected_model_success"] += int(optimized[int(np.argmin(opt))]["success"])
        if key == "one":
            totals["model_rank_agree" if int(np.argmin(opt)) == winners.index(True)
                   else "model_rank_opposite"] += 1
    print(json.dumps({"tasks": 50, "model_pairwise_ranking": ranks,
                      "physical": totals, "parity_max_action": max(x["action_diff"] for x in data["parity"]),
                      "parity_max_state": max(x["state_diff"] for x in data["parity"])},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
