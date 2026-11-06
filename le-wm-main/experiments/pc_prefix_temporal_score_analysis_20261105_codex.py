"""Summarize sampled real encoding costs without treating them as predictions."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(records):
    pairs = defaultdict(dict)
    for record in records:
        steps = record["steps"]
        if [x["step"] for x in steps] != [0, 5, 10, 15, 20, 25]:
            raise ValueError("invalid temporal sampling")
        if record["chain"] in pairs[record["index"]]:
            raise ValueError("duplicate chain")
        pairs[record["index"]][record["chain"]] = record
    if any(set(pair) != {"novel", "common"} for pair in pairs.values()):
        raise ValueError("incomplete pair")
    discordant, predicted_correct, encoded_correct = 0, 0, 0
    for pair in pairs.values():
        a, b = pair["novel"], pair["common"]
        if (a["first_success_step"] is None) == (b["first_success_step"] is None):
            continue
        discordant += 1
        winner = "novel" if a["first_success_step"] is not None else "common"
        for key, counter in (("predicted_final_cost", "pred"), ("actual_encoded_cost", "actual")):
            x = a[key] if counter == "pred" else a["steps"][-1][key]
            y = b[key] if counter == "pred" else b["steps"][-1][key]
            if x == y:
                raise ValueError("unexpected score tie")
            correct = ("novel" if x < y else "common") == winner
            if counter == "pred":
                predicted_correct += correct
            else:
                encoded_correct += correct
    ever = [r for r in records if r["first_success_step"] is not None]
    never = [r for r in records if r["first_success_step"] is None]
    return {
        "pairs": len(pairs), "chains": len(records), "ever_success": len(ever),
        "success_at_25": sum(bool(r["steps"][-1]["success_this_step"]) for r in records),
        "ever_success_but_not_at_25": sum(not r["steps"][-1]["success_this_step"] for r in ever),
        "before_25_lowest_sampled_cost": sum(min(x["actual_encoded_cost"] for x in r["steps"][:-1])
                                              < r["steps"][-1]["actual_encoded_cost"] for r in records),
        "mean_final_cost_ever_success": float(np.mean([r["steps"][-1]["actual_encoded_cost"] for r in ever])),
        "mean_final_cost_never_success": float(np.mean([r["steps"][-1]["actual_encoded_cost"] for r in never])),
        "discordant_success_pairs": discordant, "predicted_picks_success": predicted_correct,
        "actual_final_encoded_picks_success": encoded_correct,
    }


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "results/pc_prefix_temporal_score_diagnostic_dev_20261105_codex.json"
    print(json.dumps(summarize(json.loads(path.read_text())["records"]), indent=2, allow_nan=False))
