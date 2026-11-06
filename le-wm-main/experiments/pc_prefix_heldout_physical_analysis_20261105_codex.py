"""Summarize parity-gated held-out physical effects, retaining failure denominator."""
import json
from collections import Counter
from pathlib import Path


def main():
    path = (Path(__file__).resolve().parents[1] / "results" /
            "pc_prefix_heldout_physical_replay_seed42_20261105_codex.json")
    result = json.loads(path.read_text())
    if (result["errors"] or len(result["parity"]) != 50 or
            not all(item["pass"] for item in result["parity"]) or
            len(result["counterfactual"]) != 50):
        raise ValueError("failed parity or incomplete physical census")
    counts = Counter()
    legal_counts = Counter()
    for task in result["counterfactual"]:
        chains = task["chains"]
        if len(chains) != 10:
            raise ValueError("incomplete task chain set")
        parent = [[x for x in chains if x["parent"] == k] for k in range(2)]
        bs = [[x for x in p if x["chain"].startswith("b")] for p in parent]
        if any(len(x) != 4 for x in bs):
            raise ValueError("missing B chains")
        winners = [any(x["success"] for x in group) for group in bs]
        predictions = [min(x["model_cost"] for x in group) for group in bs]
        key = "both" if all(winners) else "neither" if not any(winners) else "one"
        counts[key] += 1
        counts["b_success_chains"] += sum(x["success"] for group in bs for x in group)
        counts["b_out_of_bounds_plans"] += sum(x["out_of_bounds_plan"] > 0 for group in bs for x in group)
        counts["b_out_of_bounds_executed"] += sum(x["out_of_bounds_executed"] > 0 for group in bs for x in group)
        selected = min((x for group in bs for x in group), key=lambda x: x["model_cost"])
        counts["best_predicted_b_success"] += int(selected["success"])
        counts["any_b_success"] += int(any(winners))
        if key == "one":
            counts["rank_agree" if predictions.index(min(predictions)) == winners.index(True)
                   else "rank_opposite"] += 1
        if all(x["out_of_bounds_plan"] == 0 for group in bs for x in group):
            legal_counts["tasks"] += 1
            legal_counts[key] += 1
            legal_counts["best_predicted_b_success"] += int(selected["success"])
            legal_counts["any_b_success"] += int(any(winners))
    print(json.dumps({"tasks": len(result["counterfactual"]), "all": counts,
                      "all_eight_b_plans_legal_subset": legal_counts}, indent=2))


if __name__ == "__main__":
    main()
