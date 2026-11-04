"""Validate actual feedback execution and compare paired saved old tails."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_selected_bank_first_feedback_physical_20261102 as source
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous
from . import p1_rgb_visibility_20261101_main as visibility

base, need, run, ROOT = source.base, source.need, source.run, source.ROOT
PREP_SHA = "b813af00ba8fe5472c9b9a4066d4d7dd9317897b788e512cfcb435db21cb4d7c"
REPORT_SHA = "1f2ea07f3d9227c75acc3364175e69daf4401a8052d2325aca60e9db41e469a4"


def analyze():
    directory = ROOT / "results" / f"{source.NAME}_RUN_DEV01"
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(directory / "report.json", REPORT_SHA)
    capture = base.read(ROOT / "results" / "p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["status"] == "recorded" and not report["errors"] and len(report["rows"]) == 3, "sources")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    rows = []
    for role, row in zip(spec["roles"], report["rows"], strict=True):
        need(row["arm"] == role["arm"] and row["status"] == "fixed_horizon_completed"
             and len(row["transitions"]) == 5 and len(row["observations"]) == 6, "arm")
        for action, transition in zip(role["tail"], row["transitions"], strict=True):
            need(np.array_equal(np.asarray(action, dtype="<f4"), np.asarray(transition["action"], dtype="<f4"))
                 and not transition["terminated"] and not transition["truncated"], "action or stop")
        visibility_rows = []
        for obs in row["observations"]:
            ref = {"path": str((directory / obs["file"]).resolve()), "file_sha256": obs["file_sha256"],
                   "pixel_sha256": obs["pixel_sha256"]}
            visibility_rows.append(visibility.inspect(run.coherent.base.frame(ref)))
        checkpoint = np.asarray(row["observations"][0]["state"], dtype=float)
        final = np.asarray(row["transitions"][-1]["state"], dtype=float)
        old = np.asarray(role["old_tail_final_state_evaluation_only"], dtype=float)
        need(np.array_equal(checkpoint, role["states"][-1])
             and row["observations"][0]["pixel_sha256"] == role["checkpoint"]["pixel_sha256"], "checkpoint")
        def metrics(state):
            return {"block_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
                    "block_angle_error": abs(float((state[4] - goal[4] + np.pi) % (2*np.pi) - np.pi))}
        rows.append({"arm": role["arm"], "checkpoint": metrics(checkpoint), "feedback_final": metrics(final),
                     "old_tail_final": metrics(old), "feedback_reward": row["transitions"][-1]["reward"],
                     "old_tail_reward": role["old_tail_final_reward_evaluation_only"],
                     "feedback_visibility": visibility_rows})
    return {"schema": "p1_dev305_selected_bank_first_feedback_saved_20261102.v1", "rows": rows,
            "scope": "post-outcome development; added feedback computation is not matched to old-tail baseline",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
