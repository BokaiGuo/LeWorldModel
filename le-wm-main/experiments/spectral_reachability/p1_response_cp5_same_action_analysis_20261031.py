"""Independently recompute latent scores and describe complete fixed-prefix pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_response_cp5_same_action_20261031 as source

NAME = "p1_response_cp5_same_action_analysis_20261031"
INPUT = source.ROOT / "results/p1_response_cp5_same_action_20261031_RUN01.json"
INPUT_SHA = "f52eeae3a558c1db0d034e11f07f69144ce6da1061d4cfffc8ab116496dd1f69"
METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error")


def analyze(result):
    source.need(result["schema"] == source.NAME and result["status"] == "complete" and
                result["fixed_rows"] == len(result["rows"]) + len(result["excluded"]) == 16
                and result["P1_admission"] is False, "source contract")
    rows = result["rows"]
    ids = [row["index"] for row in rows] + [row["index"] for row in result["excluded"]]
    source.need(sorted(ids) == list(range(16)), "complete unique roster")
    for row in rows:
        predicted = np.asarray(row["predicted_latent"], dtype=np.float64)
        refreshed = np.asarray(row["refreshed_latent"], dtype=np.float64)
        actual = np.asarray(row["actual_latent"], dtype=np.float64)
        goal = np.asarray(row["goal_latent"], dtype=np.float64)
        source.need(all(x.shape == (192,) and np.isfinite(x).all()
                        for x in (predicted, refreshed, actual, goal)), "latent vectors")
        for key, expected in (("predicted_goal_mse", ((predicted - goal) ** 2).mean()),
                              ("refreshed_goal_mse", ((refreshed - goal) ** 2).mean()),
                              ("actual_goal_mse", ((actual - goal) ** 2).mean()),
                              ("open_rgb_mse", ((predicted - actual) ** 2).mean()),
                              ("refreshed_rgb_mse", ((refreshed - actual) ** 2).mean())):
            source.need(abs(row[key] - float(expected)) <= 1e-6, "independent latent recomputation")
    pairs = []
    for logical in range(8):
        entries = [row for row in rows if row["logical_pair"] == logical]
        if len(entries) != 2:
            continue
        left = next(row for row in entries if row["role"] == "incumbent")
        right = next(row for row in entries if row["role"] == "model_selected")
        source.need(all(left[k] == right[k] for k in ("root_slot", "condition", "arm")),
                    "matched pair identity")
        differences = {name: right[name] - left[name] for name in
                       ("predicted_goal_mse", "refreshed_goal_mse", "actual_goal_mse")}
        physical = {name: right["physical_end"][name] - left["physical_end"][name] for name in METRICS}
        pairs.append({"logical_pair": logical, "root_slot": left["root_slot"],
                      "condition": left["condition"], "arm": left["arm"],
                      "seed_index": left["index"], "diverse_index": right["index"],
                      "diverse_minus_seed": differences,
                      "physical_endpoint_diverse_minus_seed": physical,
                      "open_vs_actual_opposite": differences["predicted_goal_mse"] *
                                                 differences["actual_goal_mse"] < 0,
                      "refreshed_vs_actual_opposite": differences["refreshed_goal_mse"] *
                                                      differences["actual_goal_mse"] < 0,
                      "actual_vs_physical_opposite": {k: differences["actual_goal_mse"] * physical[k] < 0
                                                      for k in METRICS}})
    return {"schema": NAME, "source_sha256": INPUT_SHA, "completed": len(rows),
            "excluded": result["excluded"], "pairs": pairs,
            "mean_open_rgb_mse": float(np.mean([r["open_rgb_mse"] for r in rows])),
            "mean_refreshed_rgb_mse": float(np.mean([r["refreshed_rgb_mse"] for r in rows])),
            "refreshed_lower_count": sum(r["refreshed_rgb_mse"] < r["open_rgb_mse"] for r in rows),
            "open_opposite_pairs": sum(p["open_vs_actual_opposite"] for p in pairs),
            "refreshed_opposite_pairs": sum(p["refreshed_vs_actual_opposite"] for p in pairs),
            "physical_opposite_pairs": {k: sum(p["actual_vs_physical_opposite"][k] for p in pairs)
                                         for k in METRICS},
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "Exploratory matched prefixes, not common suffix; two dependent roots; "
                           "midpoint and horizon differ; no equal-total-budget certification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    source.need(output.is_absolute() and output.parent == source.ROOT / "results" and
                output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
                "exclusive output")
    data = INPUT.read_bytes()
    source.need(hashlib.sha256(data).hexdigest() == INPUT_SHA, "input hash")
    value = analyze(json.loads(data))
    source.need(hashlib.sha256(INPUT.read_bytes()).hexdigest() == INPUT_SHA, "input drift")
    print(json.dumps({"sha256": source.common.base.write(output, value), "completed": value["completed"],
                      "pairs": len(value["pairs"]), "open_opposite_pairs": value["open_opposite_pairs"],
                      "refreshed_opposite_pairs": value["refreshed_opposite_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
