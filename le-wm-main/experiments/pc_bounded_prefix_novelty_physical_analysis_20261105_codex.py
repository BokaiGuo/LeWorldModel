"""Summarize parity-gated novelty versus same-context ordinary-pool outcomes."""
import json
from collections import Counter
from pathlib import Path


def main():
    path = (Path(__file__).resolve().parents[1] / "results" /
            "pc_bounded_prefix_novelty_physical_replay_seed42_20261105_codex.json")
    result = json.loads(path.read_text())
    if (result["errors"] or len(result["parity"]) != 50 or
            not all(item["pass"] for item in result["parity"]) or
            len(result["counterfactual"]) != 50):
        raise ValueError("physical replay did not pass for all tasks")
    counts = Counter()
    for row in result["counterfactual"]:
        if row["no_novel_candidate"]:
            counts["no_novel"] += 1
            continue
        counts["paired"] += 1
        by_name = {chain["chain"]: chain for chain in row["chains"]}
        if set(by_name) != {"novel", "common"}:
            raise ValueError("incomplete chain pair")
        novel, common = by_name["novel"], by_name["common"]
        for name, chain in by_name.items():
            counts[f"{name}_success"] += int(chain["success"])
            counts[f"{name}_illegal_plan"] += int(chain["out_of_bounds_plan"] > 0)
            counts[f"{name}_illegal_executed"] += int(chain["out_of_bounds_executed"] > 0)
        counts["only_novel_success"] += int(novel["success"] and not common["success"])
        counts["only_common_success"] += int(common["success"] and not novel["success"])
        counts["novel_model_lower"] += int(row["model_cost_gap"] < 0)
        counts["novel_model_lower_only_novel_success"] += int(
            row["model_cost_gap"] < 0 and novel["success"] and not common["success"])
    print(json.dumps({"tasks": 50, "results": counts,
                      "parity_steps": sum(x["steps"] for x in result["parity"]),
                      "parity_max_action_diff": max(x["action_diff"] for x in result["parity"]),
                      "parity_max_state_diff": max(x["state_diff"] for x in result["parity"])},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
