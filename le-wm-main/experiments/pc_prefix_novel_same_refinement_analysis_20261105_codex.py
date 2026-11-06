"""Summarize parity-gated paired physical outcomes before and after refinement."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(root, name):
    data = json.loads((root / name).read_text())
    if (data["errors"] or len(data["parity"]) != 50 or
            not all(x["pass"] for x in data["parity"]) or
            len(data["counterfactual"]) != 50):
        raise ValueError(f"failed parity or missing records: {name}")
    return data


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    before = load(root, "pc_bounded_prefix_novelty_physical_replay_seed42_20261105_codex.json")
    after = load(root, "pc_prefix_novel_same_refinement_replay_seed42_20261105_codex.json")
    original_path = root / "pc_bounded_prefix_full_pool_novelty_seed42_20261105_codex.audit.jsonl"
    refined_path = root / "pc_prefix_novel_same_refinement_seed42_20261105_codex.audit.jsonl"
    original = json.loads(original_path.read_text().splitlines()[0])
    refined = json.loads(refined_path.read_text().splitlines()[0])
    pairs = {}
    for iteration in original["iterations"]:
        audit = iteration["full_pool_audit"]
        if audit is None:
            continue
        for offset, item in enumerate(audit["records"]):
            index = iteration["batch_start"] + offset
            candidate = item["best_unique"]
            if candidate is not None and (index not in pairs or
                    candidate["cost_minus_pool_best"] < pairs[index]["best_unique"]["cost_minus_pool_best"]):
                pairs[index] = item
    for row in refined["refinement"]:
        index = row["index"]
        if row["no_novel_candidate"]:
            if index in pairs:
                raise ValueError("novel presence changed")
            continue
        pair = pairs[index]
        expected = [pair["best_unique"]["actions"], pair["pool_best_actions"]]
        if not np.array_equal(np.asarray(expected), np.asarray(row["original_actions"])):
            raise ValueError(f"refinement starts differ for task {index}")
    counts = Counter()
    for a, b in zip(before["counterfactual"], after["counterfactual"]):
        if a["index"] != b["index"] or a["no_novel_candidate"] != b["no_novel_candidate"]:
            raise ValueError("task pairing differs")
        if a["no_novel_candidate"]:
            counts["no_novel"] += 1
            continue
        pre = {chain["chain"]: chain for chain in a["chains"]}
        post = {chain["chain"]: chain for chain in b["chains"]}
        if set(pre) != {"novel", "common"} or set(post) != set(pre):
            raise ValueError("incomplete chain pair")
        counts["paired"] += 1
        for name in ("novel", "common"):
            counts[f"{name}_before_success"] += int(pre[name]["success"])
            counts[f"{name}_after_success"] += int(post[name]["success"])
            counts[f"{name}_gained"] += int(not pre[name]["success"] and post[name]["success"])
            counts[f"{name}_lost"] += int(pre[name]["success"] and not post[name]["success"])
            counts[f"{name}_illegal_plan"] += int(post[name]["out_of_bounds_plan"] > 0)
        counts["after_only_novel_success"] += int(post["novel"]["success"] and not post["common"]["success"])
        counts["after_only_common_success"] += int(post["common"]["success"] and not post["novel"]["success"])
        counts["refined_novel_lower_cost"] += int(b["model_cost_gap"] < 0)
        counts["extra_model_scores"] += b["refinement_scores"]
    print(json.dumps({"tasks": 50, "counts": counts,
                      "parity_steps": sum(x["steps"] for x in after["parity"]),
                      "max_state_diff": max(x["state_diff"] for x in after["parity"]),
                      "max_action_diff": max(x["action_diff"] for x in after["parity"])},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
