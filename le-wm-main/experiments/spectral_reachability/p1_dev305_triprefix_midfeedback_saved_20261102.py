"""Validate saved three-prefix feedback execution without rerunning inference."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_triprefix_midfeedback_physical_20261102 as source

base, need, run, ROOT = source.base, source.need, source.run, source.ROOT
CONTRACT_SHA = "e6152cb5126a9d990d5e58171035a93569223306bba64ae607be7d95dfa95b72"
REPORT_SHA = "6ab1e0e45876c1c25d3ede288bc41c5a6ee630ba710220b2db701d22ff81cd69"


def analyze():
    directory = ROOT / "results" / f"{source.NAME}_RUN_DEV01"
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(directory / "report.json", REPORT_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["status"] == "recorded" and not report["errors"] and len(report["rows"]) == 3, "sources")
    rows = []
    for role, row in zip(spec["roles"], report["rows"], strict=True):
        need(row["arm"] == role["arm"] and row["status"] == "fixed_horizon_completed"
             and len(row["transitions"]) == 5 and len(row["observations"]) == 6, "arm")
        for action, transition in zip(role["tail"], row["transitions"], strict=True):
            need(np.array_equal(np.asarray(action, dtype="<f4"), np.asarray(transition["action"], dtype="<f4"))
                 and not transition["terminated"] and not transition["truncated"], "action or stop")
        for obs in row["observations"]:
            ref = {"path": str((directory / obs["file"]).resolve()), "file_sha256": obs["file_sha256"],
                   "pixel_sha256": obs["pixel_sha256"]}
            run.coherent.base.frame(ref)
        checkpoint = np.asarray(row["observations"][0]["state"], dtype=float)
        final = np.asarray(row["transitions"][-1]["state"], dtype=float)
        old = np.asarray(role["old_tail_final_state_evaluation_only"], dtype=float)
        need(np.array_equal(checkpoint, role["states"][-1])
             and row["observations"][0]["pixel_sha256"] == role["checkpoint"]["pixel_sha256"], "checkpoint")
        rows.append({"arm": role["arm"], "status": row["status"],
                     "checkpoint_state": checkpoint.tolist(), "feedback_final_state": final.tolist(),
                     "old_tail_final_state": old.tolist(),
                     "feedback_reward": row["transitions"][-1]["reward"],
                     "old_tail_reward": role["old_tail_final_reward_evaluation_only"],
                     "block_pose_changed_after_feedback": bool(not np.array_equal(checkpoint[2:5], final[2:5])),
                     "block_pose_changed_in_old_tail": bool(not np.array_equal(checkpoint[2:5], old[2:5]))})
    return {"schema": "p1_dev305_triprefix_midfeedback_saved_20261102.v1", "rows": rows,
            "scope": "known-root development; plan selection and neural inference not independently recomputed",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
