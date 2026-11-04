"""Verify cp10 feedback plans, replay checkpoints, and saved physical outcomes."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev304_spectral_cp10_physical_20261102 as physical
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, planner = physical.base, physical.planner
ROOT = physical.ROOT/"results"
RUN = ROOT/f"{physical.NAME}_RUN_DEV01"
CONTRACT_SHA = "07b315954b1cd951cce758a3802b527f8b2116f5c26f365cbcc26e8983d0d9a0"
REPORT_SHA = "9060a4909c7e02a985528d0015cc0bf399491d079180e16c47ef5be930945c7f"
need = physical.need


def analyze():
    contract = base.read(ROOT/f"{physical.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(RUN/"report.json", REPORT_SHA)
    plan = base.read(ROOT/f"{planner.NAME}_PLAN_DEV01/report.json", physical.PLAN_SHA)
    need(contract == physical.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["status"] == "recorded" and not report["errors"]
         and base.sha(RUN/"journal.jsonl") == report["journal_sha256"], "source")
    capture = base.read(planner.saved.ROOT/f"{planner.saved.NAME}_CAPTURE_DEV01/report.json",
                        planner.saved.CAPTURE_SHA)
    goal = capture["goal_state_evaluation_only"]
    need(len(report["rows"]) == len(contract["roles"]) == len(plan["rows"]) == 4, "roster")
    rows = []
    for role, row, planned in zip(contract["roles"], report["rows"], plan["rows"], strict=True):
        need(role["arm"] == row["arm"] == planned["arm"]
             and base.read(RUN/(row["arm"]+".row.json")) == row
             and row["status"] == "fixed_horizon_completed", "row or termination")
        need(len(row["transitions"]) == 5 and len(row["observations"]) == 6, "length")
        for i, (transition, action) in enumerate(zip(row["transitions"], role["tail"], strict=True), 1):
            need(transition["step"] == i and np.array_equal(transition["action"], action)
                 and not transition["terminated"] and not transition["truncated"], "tail")
        for obs in row["observations"]:
            planner.saved.verify_frame(RUN, obs)
        need(all(row["observations"][0][k] == role["checkpoint"][k]
                 for k in ("pixel_sha256", "state", "snapshot")), "cp10")
        need(all(o["state"] == t["state"] for o, t in zip(row["observations"][1:],
                                                           row["transitions"], strict=True)), "frame states")
        rows.append({"arm": row["arm"], "reward": row["transitions"][-1]["reward"],
                     "terminal": measures(row["transitions"][-1]["state"], goal),
                     "native_success": False})
    for role in contract["roles"]:
        for module in ("action_encoder", "predictor"):
            need(plan["ledger"][role["arm"]+":"+module] == {"calls": 10, "samples": 3200}, "ledger")
    return {"schema": physical.NAME+".saved.v1", "rows": rows, "frames_verified": 24,
            "classification": "one_known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
