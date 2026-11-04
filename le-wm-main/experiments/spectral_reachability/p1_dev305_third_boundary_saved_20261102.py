"""Revalidate third-boundary selection, action dispatch, and physical frame hashes."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_third_boundary_physical_20261102 as physical

base, need, ROOT = physical.base, physical.need, physical.ROOT
NAME = "p1_dev305_third_boundary_saved_20261102"
PREP_SHA = "af81c0c4ac11dbee0db6500d8cc458c0bee13c6f4776d69754ab744e3a9284a7"
REPORT_SHA = "83d8341425938f7b39b24f4f93c2b4e81a9efcd5b5d4fdd117b34f26e0636d02"


def analyze():
    spec = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{physical.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(spec == physical.protocol() and report["contract_sha256"] == PREP_SHA
         and report["status"] == "recorded" and not report["errors"]
         and len(spec["roles"]) == len(report["rows"]) == 6, "contract")
    directory = ROOT / "results" / f"{physical.NAME}_RUN_DEV01"
    rows = []
    for role, result in zip(spec["roles"], report["rows"], strict=True):
        need(role["arm"] == result["arm"] and role["selected"] == result["selected"], "role")
        transitions, frames = result["transitions"], result["observations"]
        need(len(frames) == len(transitions) + 1 and len(transitions) <= 5, "length")
        visible = []
        for step, frame in enumerate(frames):
            need(frame["step"] == step, "step")
            ref = {"path": str((directory / frame["file"]).resolve()),
                   "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
            pixels = physical.planner.run.coherent.base.frame(ref)
            visible.append(physical.risk.engine.risk.visibility.inspect(pixels)["eligible"])
        need(frames[0]["pixel_sha256"] == role["checkpoint"]["pixel_sha256"]
             and visible[-1] == result["final_visibility"]["eligible"], "frame binding")
        for i, step in enumerate(transitions):
            need(step["step"] == i + 1 and visible[i]
                 and np.array_equal(np.asarray(step["action"], dtype="<f4"),
                                    np.asarray(role["tail"][i], dtype="<f4")), "dispatch")
        need(result["status"] == "fixed_horizon_completed" and len(transitions) == 5, "completion")
        rows.append({"arm": role["arm"], "candidate": role["candidate"], "selected": role["selected"],
                     "all_frames_visible": all(visible), "final_visible": visible[-1],
                     "reward": transitions[-1]["reward"], "success": bool(transitions[-1]["terminated"])})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "known development root, not independent confirmation or a complete fair-budget study",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
