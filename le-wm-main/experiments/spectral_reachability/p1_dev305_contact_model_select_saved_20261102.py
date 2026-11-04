"""Verify the contact packet physical receipts and report terminal task metrics."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_contact_model_select_physical_20261102 as source
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous
from . import p1_rgb_visibility_20261101_main as visibility

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "54d64056eee1082be2da7a237107ac6adf4559282105889972b4d1e9535ce558"
REPORT_SHA = "729626023ac4668eb1e07c570e58363ef77081c02bb47bd8b6df40d878a03151"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    capture = base.read(ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and not report["errors"] and len(report["rows"]) == len(spec["roles"]) == 12, "binding")
    goal = np.asarray(capture["goal_state_evaluation_only"])
    directory = ROOT / "results" / f"{source.NAME}_RUN_DEV01"
    rows = []
    for role, result in zip(spec["roles"], report["rows"], strict=True):
        need(role["arm"] == result["arm"] and result["status"] == "fixed_horizon_completed"
             and len(result["transitions"]) == 5 and len(result["observations"]) == 6, "execution")
        expected = np.asarray(role["tail"], dtype="<f4")
        actual = np.asarray([x["action"] for x in result["transitions"]], dtype="<f4")
        need(np.array_equal(expected, actual), "actions")
        initial = np.asarray(result["observations"][0]["state"])
        final = np.asarray(result["transitions"][-1]["state"])
        screens = []
        for obs in result["observations"]:
            ref = {"path": str((directory / obs["file"]).resolve()), "file_sha256": obs["file_sha256"],
                   "pixel_sha256": obs["pixel_sha256"]}
            screens.append(visibility.inspect(source.planner.run.coherent.base.frame(ref))["eligible"])
        rows.append({"arm": role["arm"], "selected": role["selected"], "reward": result["transitions"][-1]["reward"],
                     "object_displacement": float(np.linalg.norm(final[2:4] - initial[2:4])),
                     "object_angle_delta": float((final[4] - initial[4] + np.pi) % (2*np.pi) - np.pi),
                     "object_position_error": float(np.linalg.norm(final[2:4] - goal[2:4])),
                     "object_angle_error": abs(float((final[4] - goal[4] + np.pi) % (2*np.pi) - np.pi)),
                     "pusher_position_error": float(np.linalg.norm(final[:2] - goal[:2])),
                     "visibility_all_six_frames": all(screens), "visibility_final": screens[-1],
                     "native_success": bool(result["transitions"][-1]["terminated"])})
    return {"schema": "p1_dev305_contact_model_select_saved_20261102.v1", "rows": rows,
            "scope": "known root, fixed RGB-only geometric candidate family, model-selected; not D2P or independent confirmation",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
