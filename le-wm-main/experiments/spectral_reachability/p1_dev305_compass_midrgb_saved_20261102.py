"""Recompute descriptive same-action refresh results from the sealed report."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_compass_midrgb_model_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
CONTRACT_SHA = "cc8fa593bdf176cda58edcca63861d9ff7c223b2809960c45a8fe04e053bcb91"
REPORT_SHA = "d3483151eaf5d9cf24451ccaabc21a17ee3da631cd529c515a55b9e80e14383b"


def pair_counts(rows, score, tolerance=0.0):
    result = {"same": 0, "reversed": 0, "ties": 0}
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            x = a[score] - b[score]
            y = a["actual_goal_mse"] - b["actual_goal_mse"]
            if abs(x) <= tolerance or abs(y) <= tolerance:
                result["ties"] += 1
            elif x * y > 0:
                result["same"] += 1
            else:
                result["reversed"] += 1
    return result


def analyze():
    contract = base.read(ROOT / f"results/{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / f"results/{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"], "provenance")
    expected = [(role, item) for role in contract["roles"] for item in role["rows"]]
    need(len(expected) == len(report["rows"]) == 54, "roster")
    for (role, item), row in zip(expected, report["rows"], strict=True):
        need(row["arm"] == role["arm"] and row["name"] == item["name"]
             and row["open_goal_mse"] == item["open_goal_mse"]
             and row["open_prediction_mse"] == item["open_prediction_mse"]
             and np.isclose(row["actual_goal_mse"], item["actual_goal_mse"], atol=2e-6, rtol=0)
             and all(np.isfinite(row[key]) and row[key] >= 0 for key in
                     ("open_goal_mse", "actual_goal_mse", "refreshed_goal_mse",
                      "open_prediction_mse", "refreshed_prediction_mse")), "row binding")
    summaries = []
    for role in contract["roles"]:
        rows = [row for row in report["rows"] if row["arm"] == role["arm"]]
        summaries.append({"arm": role["arm"], "count": len(rows),
                          "open_prediction_mse_mean": float(np.mean([r["open_prediction_mse"] for r in rows])),
                          "refreshed_prediction_mse_mean": float(np.mean([r["refreshed_prediction_mse"] for r in rows])),
                          "refresh_lower_count": sum(r["refreshed_prediction_mse"] < r["open_prediction_mse"] for r in rows),
                          "open_pairs_exact": pair_counts(rows, "open_goal_mse"),
                          "refreshed_pairs_exact": pair_counts(rows, "refreshed_goal_mse"),
                          "open_pairs_tolerance_1e-6": pair_counts(rows, "open_goal_mse", 1e-6),
                          "refreshed_pairs_tolerance_1e-6": pair_counts(rows, "refreshed_goal_mse", 1e-6)})
    return {"schema": "p1_dev305_compass_midrgb_saved_20261102.v1", "rows": summaries,
            "scope": "same known root, shared trajectories, descriptive post-outcome, no causal control benefit",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
