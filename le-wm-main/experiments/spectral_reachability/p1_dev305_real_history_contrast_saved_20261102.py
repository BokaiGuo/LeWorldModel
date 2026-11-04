"""Validate and summarize matched real-history prediction contrast."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_real_history_contrast_20261102 as source

base, need = source.base, source.need
ROOT = source.ROOT / "results"
CONTRACT_SHA = "a78543239d5de4aea4a68af69ea2201ed0ee5d63c3845bd085404b49f241df3d"
REPORT_SHA = "73105a010126057811f6cac0a1c0d2d39e774a5f4cb8d2e715717b5c4130c228"


def analyze():
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    previous = base.read(ROOT / f"{source.prior.NAME}_RUN_DEV01/report.json", source.PRIOR_REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"], "source")
    expected = [(role, row) for role in contract["roles"] for row in role["rows"]]
    need(len(expected) == len(report["rows"]) == len(previous["rows"]) == 54, "roster")
    for (role, expected_row), row, prior in zip(expected, report["rows"], previous["rows"], strict=True):
        need(row["arm"] == prior["arm"] == role["arm"]
             and row["name"] == prior["name"] == expected_row["name"]
             and row["actions_sha256"] == prior["actions_sha256"] == expected_row["actions_sha256"]
             and np.isclose(row["true_goal_mse"], prior["actual_goal_mse"], atol=2e-6)
             and np.isclose(row["real_history"]["goal_mse"], prior["predicted_goal_mse"], atol=2e-6)
             and np.isclose(row["real_history"]["prediction_mse"], prior["prediction_mse"], atol=2e-6)
             and all(np.isfinite(row[mode][metric]) and row[mode][metric] >= 0
                     for mode in ("real_history", "single_frame")
                     for metric in ("prediction_mse", "goal_mse")), "row consistency")
    summaries = []
    for role in contract["roles"]:
        rows = [r for r in report["rows"] if r["arm"] == role["arm"]]
        real = np.array([r["real_history"]["prediction_mse"] for r in rows])
        single = np.array([r["single_frame"]["prediction_mse"] for r in rows])
        ordering = {}
        for mode in ("real_history", "single_frame"):
            consistent = reversed_ = ties = 0
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    pred = np.sign(a[mode]["goal_mse"] - b[mode]["goal_mse"])
                    truth = np.sign(a["true_goal_mse"] - b["true_goal_mse"])
                    if pred == 0 or truth == 0:
                        ties += 1
                    elif pred == truth:
                        consistent += 1
                    else:
                        reversed_ += 1
            ordering[mode] = {"same": consistent, "reversed": reversed_, "ties": ties}
        summaries.append({"arm": role["arm"], "real_history_mse_mean": float(real.mean()),
                          "single_frame_mse_mean": float(single.mean()),
                          "real_history_lower_count": int((real < single).sum()),
                          "real_history_higher_count": int((real > single).sum()), "ordering": ordering})
    return {"schema": source.NAME + ".analysis.v1", "roles": summaries,
            "scope": "one development root; no tolerance for ordering; not independent candidate samples",
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
