"""Verify every recorded feedback action from its saved RGB and report outcomes."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_contact_rgb_feedback_20261102 as source
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "8403b84f4f33208f9e5a7f03b591457bbfb2c81bd2a888e383985774de48bb28"
REPORT_SHA = "27ad354e1c47774c940f6f86a724d4a1828fc96ac5c6deadbfc775d43c576650"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    capture = base.read(ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    need(report.get("status") == "recorded" and report["errors"] == [], "report status/errors")
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and len(report["rows"]) == len(spec["roles"]) == 9, "binding")
    goal = np.asarray(capture["goal_state_evaluation_only"])
    directory = ROOT / "results" / f"{source.NAME}_RUN_DEV01"
    rows = []
    for role, entry in zip(spec["roles"], report["rows"], strict=True):
        need(role["arm"] == entry["arm"] and entry["status"] in
             ("fixed_horizon_completed", "native_early_stop", "visibility_guard_stop"), "role")
        obs, transitions = entry["observations"], entry["transitions"]
        need(len(obs) == len(transitions) + 1 and 1 <= len(transitions) <= 5, "length")
        need((entry["status"] == "fixed_horizon_completed") == (len(transitions) == 5)
             or (entry["status"] == "native_early_stop" and len(transitions) == 5),
             "status and horizon")
        if entry["status"] == "native_early_stop":
            need(transitions[-1]["terminated"] or transitions[-1]["truncated"],
                 "native stop requires returned transition")
        pixels = []
        for frame in obs:
            ref = {"path": str((directory / frame["file"]).resolve()), "file_sha256": frame["file_sha256"],
                   "pixel_sha256": frame["pixel_sha256"]}
            pixels.append(source.prior.rgb_source.run.coherent.base.frame(ref))
        for index, transition in enumerate(transitions):
            need(source.prior.rgb_source.visibility.inspect(pixels[index])["eligible"], "guard")
            geometry = source.prior.rgb_source.run.coherent.base.first.engine._geometry(pixels[index])
            action = source.choose_action(role["mode"], geometry["contact_direction_raw_xy"], role["initial_direction"])
            need(np.array_equal(action, np.asarray(transition["action"], dtype="<f4"))
                 and np.allclose(geometry["contact_direction_raw_xy"], transition["direction_from_rgb"], atol=1e-12)
                 and transition["step"] == index + 1, "rgb/action binding")
        last_visible = source.prior.rgb_source.visibility.inspect(pixels[-1])["eligible"]
        need(last_visible == entry["final_visibility"]["eligible"], "final guard")
        if entry["status"] == "visibility_guard_stop":
            need(not last_visible and len(transitions) < 5, "stop rule")
        initial = np.asarray(obs[0]["state"])
        final = np.asarray(transitions[-1]["state"])
        rows.append({"arm": role["arm"], "status": entry["status"], "steps": len(transitions),
                     "visible_final": bool(last_visible),
                     "environment_contact_counter_positive_steps": sum(x["native_contact_points"] > 0 for x in transitions),
                     "object_displacement": float(np.linalg.norm(final[2:4] - initial[2:4])),
                     "object_position_error": float(np.linalg.norm(final[2:4] - goal[2:4])),
                     "object_angle_error": abs(float((final[4] - goal[4] + np.pi) % (2*np.pi) - np.pi)),
                     "reward": transitions[-1]["reward"],
                     "native_success": (None if entry["status"] == "visibility_guard_stop"
                                        else bool(transitions[-1]["terminated"]))})
    return {"schema": "p1_dev305_contact_rgb_feedback_saved_20261102.v1", "rows": rows,
            "scope": "same known root; geometry control; guard is reactive; counter not certified object-pusher contact",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
