"""Pair saved one-packet predictions with their exactly replayed physical tails."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_cp10_tail_pair_cuda_20261102 as cuda

base, need = cuda.base, cuda.need
ROOT = cuda.ROOT/"results"
CONTRACT_SHA = "da830bd8a7bc94f843f877b5bb3243fbb010e4bf50f176ca9ad56bb55ab814f2"
REPORT_SHA = "cfb4c15e59093b5b81630a3fcee9908832fadff1fbc64c5f461957c6d4b87ce4"
RUN = ROOT/f"{cuda.NAME}_RUN_DEV01"


def analyze():
    spec = base.read(ROOT/f"{cuda.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(RUN/"report.json", REPORT_SHA)
    need(spec == cuda.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and report["counts"]["predictor"] == {"calls": 2, "samples": 4}, "source")
    new = {row["arm"]: row for row in cuda.verified.analyze()["rows"]}
    old_panels = {"goal": cuda.planner.goal_saved.analyze()["rows"]}
    rows = []
    for role, predicted in zip(spec["roles"], report["rows"], strict=True):
        need(role["arm"] == predicted["arm"] and base.read(RUN/(role["arm"]+".json")) == predicted,
             "row binding")
        label, prefix = role["arm"].split("_", 1)
        old = next(r for r in old_panels[label] if r["prefix"] == prefix and r["selected"])
        actual = new[role["arm"]]
        values = [predicted["old_score"], predicted["new_score"], old["reward"], actual["reward"]]
        need(np.isfinite(values).all(), "finite")
        rows.append({"arm": role["arm"], "old_score": values[0], "new_score": values[1],
                     "old_reward": values[2], "new_reward": values[3],
                     "old_position_error": old["terminal"]["object_position_error"],
                     "new_position_error": actual["terminal"]["object_position_error"],
                     "old_angle_error": old["terminal"]["object_angle_error"],
                     "new_angle_error": actual["terminal"]["object_angle_error"],
                     "model_prefers_new": values[1] < values[0],
                     "reward_prefers_new": values[3] > values[2]})
    return {"schema": cuda.NAME+".analysis.v1", "rows": rows,
            "classification": "post_outcome_one_known_root", "p1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(base.write_new(p.parse_args().output, analyze()))
