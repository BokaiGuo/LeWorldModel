"""Validate sealed second-boundary physical branches and action/frame alignment."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_contact_second_boundary_physical_20261102 as physical

base, need, ROOT = physical.base, physical.need, physical.ROOT
CONTRACT_SHA = "8276ce7b04b98fa165793bce4374919aacc98080f79718ab6c2121d285c32e8b"
REPORT_SHA = "7296e02b828ffa3fb110429da914fc1b2726c5c36c1b20c2c9ac998aea2cec4e"
NAME = "p1_dev305_contact_second_boundary_saved_20261102"


def analyze():
    contract = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / "results" / f"{physical.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    need(contract == physical.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["status"] == "recorded" and not report["errors"]
         and len(contract["roles"]) == len(report["rows"]) == 12, "sources")
    directory = ROOT / "results" / f"{physical.NAME}_RUN_DEV01"
    rows = []
    for spec, row in zip(contract["roles"], report["rows"], strict=True):
        need(spec["arm"] == row["arm"] and spec["selected"] == row["selected"], "arm")
        transitions = row["transitions"]
        observations = row["observations"]
        need(len(observations) == len(transitions) + 1 and len(transitions) <= 5, "length")
        frames = []
        for step, item in enumerate(observations):
            need(item["step"] == step, "frame index")
            ref = {"path": str((directory / item["file"]).resolve()),
                   "file_sha256": item["file_sha256"], "pixel_sha256": item["pixel_sha256"]}
            frame = physical.planner.run.coherent.base.frame(ref)
            visible = physical.planner.previous.source.visibility.inspect(frame)["eligible"]
            frames.append(visible)
        need(observations[0]["pixel_sha256"] == spec["checkpoint"]["pixel_sha256"], "checkpoint")
        for i, item in enumerate(transitions):
            need(item["step"] == i + 1 and frames[i]
                 and np.array_equal(np.asarray(item["action"], dtype="<f4"),
                                    np.asarray(spec["tail"][i], dtype="<f4")), "dispatch")
        need(frames[-1] == row["final_visibility"]["eligible"], "final frame")
        if row["status"] == "visibility_guard_stop":
            need(len(transitions) < 5 and not frames[-1], "guard stop")
        elif row["status"] == "fixed_horizon_completed":
            need(len(transitions) == 5, "completed")
        elif row["status"] == "native_early_stop":
            need(bool(transitions) and (transitions[-1]["terminated"] or transitions[-1]["truncated"]),
                 "native stop")
        else:
            need(False, "unexpected status")
        rows.append({"arm": row["arm"], "selected": row["selected"], "status": row["status"],
                     "steps": len(transitions), "final_visible": frames[-1],
                     "success": bool(transitions and transitions[-1]["terminated"]),
                     "reward": transitions[-1]["reward"] if transitions else None})
    return {"schema": NAME + ".v1", "rows": rows,
            "scope": "single known development root; different stop times are censored; not full model or planning costs",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
