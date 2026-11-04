"""Verify coherent-seed simulated trajectories against the fixed CEM panel."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_cp10_coherent_physical_20261102 as physical
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

ROOT = physical.ROOT
RUN = ROOT/"results/p1_cp10_coherent_physical_20261102_RUN_DEV01"
CONTRACT = ROOT/"results/p1_cp10_coherent_physical_20261102_PREP_DEV01.json"
CONTRACT_SHA = "4c01d95e6b8ab831f76557e1d4e8ef30da567c43b222b7de9458f372c1f791b0"
REPORT_SHA = "6108ec7a6d922331c939ce564bc26e610e61c5a23d00e718a2db6fb1e5546a29"
need = physical.require


def analyze():
    contract = physical.base.read(CONTRACT, CONTRACT_SHA)
    need(contract == physical.protocol(), "contract closure")
    report = physical.base.read(RUN/"report.json", REPORT_SHA)
    prior = physical.base.read(physical.previous.RUN/"report.json", physical.previous.REPORT_SHA)
    need(report["status"] == "recorded" and not report["errors"]
         and report["prepare_sha256"] == CONTRACT_SHA
         and physical.base.sha(RUN/"journal.jsonl") == report["journal_sha256"], "report and journal")
    need([r["arm"] for r in report["rows"]] == [r["arm"] for r in contract["roles"]]
         == [r["arm"] for r in prior["rows"]], "roster")
    rows, frames = [], 0
    for role, row, old in zip(contract["roles"], report["rows"], prior["rows"], strict=True):
        need(physical.base.read(RUN/(role["arm"]+".row.json")) == row
             and row["status"] == old["status"] == "fixed_horizon_completed", "row or censoring")
        actions = ([("warmup", i+1, [0., 0.]) for i in range(10)]
                   + [("prefix", i+1, a) for i, a in enumerate(role["prefix"])]
                   + [("old_tail", i+1, a) for i, a in enumerate(role["old_first_packet"])]
                   + [("coherent_tail", i+1, a) for i, a in enumerate(role["new_first_packet"])])
        need(len(row["transitions"]) == len(actions), "transition count")
        for t, (phase, step, action) in zip(row["transitions"], actions, strict=True):
            need(t["phase"] == phase and t["step"] == step and np.array_equal(t["action"], action)
                 and not t["terminated"] and not t["truncated"], "action/terminal binding")
        roster = ([("warmup", i) for i in (0, 5, 10)] + [("prefix", i) for i in range(1, 6)]
                  + [("tail", 5)] + [("coherent_tail", i) for i in range(1, 6)])
        need([(o["phase"], o["step"]) for o in row["observations"]] == roster, "frame roster")
        for obs in row["observations"]:
            path = RUN/obs["file"]
            need(path.parent == RUN and not path.is_symlink()
                 and physical.base.sha(path) == obs["file_sha256"], "frame file")
            with Image.open(path) as img:
                pixels = np.asarray(img.convert("RGB"))
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == obs["pixel_sha256"], "frame pixels")
            frames += 1
        cp10 = next(o for o in row["observations"] if o["phase"] == "tail")
        need(all(cp10[k] == role["cp10_checkpoint"][k] for k in ("pixel_sha256", "state", "snapshot")),
             "cp10 snapshot")
        need(all(np.array_equal(x["action"], y["action"]) and x["state"] == y["state"]
                 for x, y in zip(row["transitions"][:20], old["transitions"][:20], strict=True)),
             "paired pre-feedback replay")
        new, prior_measures = (measures(r["transitions"][-1]["state"], contract["goal_state"]) for r in (row, old))
        rows.append({"arm": role["arm"], "coherent": new, "original": prior_measures,
                     "coherent_minus_original": {k: new[k]-prior_measures[k] for k in new},
                     "coherent_reward": row["transitions"][-1]["reward"],
                     "original_reward": old["transitions"][-1]["reward"],
                     "coherent_success": False, "original_success": False})
    return {"schema": "p1_cp10_coherent_saved.v1", "rows": rows, "frames_verified": frames,
            "source_report_sha256": REPORT_SHA, "classification": "known_development_root_only",
            "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(physical.base.write_new(a.output, analyze()))
