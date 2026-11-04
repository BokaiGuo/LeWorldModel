"""Audit and summarize saved fixed-bank prediction and real execution comparisons."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_compass_model_compare_20261102 as source

base, need = source.base, source.need
ROOT = source.ROOT / "results"
CONTRACT_SHA = "cbb740dac63a72016b222e83b029def2d783e0ec6cd9183e2ecc237d88743339"
REPORT_SHA = "8386fdd538665362a0c476c757d45d70fbc779b091ac3a53128294400e755b96"


def analyze():
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"], "source")
    expected = [(role, row) for role in contract["roles"] for row in role["rows"]]
    need(len(expected) == len(report["rows"]) == 54, "complete roster")
    for (role, expected_row), row in zip(expected, report["rows"], strict=True):
        need(row["arm"] == role["arm"] and row["name"] == expected_row["name"]
             and row["actions_sha256"] == expected_row["actions_sha256"]
             and row["guard_dot"] == expected_row["guard_dot"]
             and row["feasible"] == expected_row["feasible"]
             and row["reward_evaluation_only"] == expected_row["reward_evaluation_only"]
             and all(np.isfinite(row[k]) for k in ("actual_goal_mse", "predicted_goal_mse", "prediction_mse"))
             and row["prediction_mse"] >= 0, "row binding")
    result = []
    for role in contract["roles"]:
        rows = [r for r in report["rows"] if r["arm"] == role["arm"]]
        for left, right in (("predicted_goal_mse", "actual_goal_mse"),
                            ("actual_goal_mse", "reward_evaluation_only")):
            same, reversed_, tied = 0, 0, 0
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    x = np.sign(a[left] - b[left])
                    y = np.sign(a[right] - b[right]) * (-1 if right == "reward_evaluation_only" else 1)
                    if x == 0 or y == 0:
                        tied += 1
                    elif x == y:
                        same += 1
                    else:
                        reversed_ += 1
            result.append({"arm": role["arm"], "comparison": left + "_vs_" + right,
                           "same": same, "reversed": reversed_, "ties": tied})
    return {"schema": source.NAME + ".analysis.v1", "pair_counts": result,
            "scope": "descriptive same-root same-action, no tie tolerance, no independent inference",
            "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
