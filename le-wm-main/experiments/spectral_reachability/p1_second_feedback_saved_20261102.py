"""Verify and compare actual cp10 feedback against the sealed old continuation."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_second_feedback_physical_20261102 as physical
from . import p1_residual_cp5_timing_saved_20261102 as old
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

ROOT = physical.ROOT
CONTRACT = ROOT / "results/p1_second_feedback_physical_20261102_PREP_DEV01.json"
CONTRACT_SHA = "45efb461b5c3e6eda87e1b8857609701800339eda2fb42d4a2358d4876c8ace4"
RUN = ROOT / "results/p1_second_feedback_physical_20261102_RUN_DEV01"
REPORT_SHA = "967ae167f490e1e6d829aedeec4a3b59f6305903f6fc0f5fc94c22a78ed610e1"
need = physical.require


def analyze():
    contract = physical.previous.read(CONTRACT, CONTRACT_SHA)
    need(contract == physical.build(physical.MODEL_SHA), "contract closure")
    report = physical.previous.read(RUN / "report.json", REPORT_SHA)
    original = physical.previous.read(old.RUN / "report.json", old.REPORT_SHA)
    need(report["status"] == "recorded" and not report["errors"]
         and report["prepare_sha256"] == CONTRACT_SHA
         and physical.previous.sha(RUN / "journal.jsonl") == report["journal_sha256"], "run integrity")
    need([r["arm"] for r in report["rows"]] == [r["arm"] for r in contract["roles"]]
         == [r["arm"] for r in original["rows"]], "roster")
    rows, frames = [], 0
    for role, row, prior in zip(contract["roles"], report["rows"], original["rows"], strict=True):
        need(physical.previous.read(RUN / (role["arm"] + ".row.json")) == row
             and row["status"] == "fixed_horizon_completed", "row shard or censoring")
        expected = ([("warmup", i + 1, [0., 0.]) for i in range(10)]
                    + [("prefix", i + 1, a) for i, a in enumerate(role["prefix"])]
                    + [("old_tail", i + 1, a) for i, a in enumerate(role["previous_first_packet"])]
                    + [("new_tail", i + 1, a) for i, a in enumerate(role["new_first_packet"])])
        need(len(row["transitions"]) == len(expected), "transition count")
        for t, (phase, step, action) in zip(row["transitions"], expected, strict=True):
            need(t["phase"] == phase and t["step"] == step
                 and np.array_equal(t["action"], action)
                 and not t["terminated"] and not t["truncated"], "transition mismatch")
        expected_frames = ([("warmup", i) for i in (0, 5, 10)]
                           + [("prefix", i) for i in range(1, 6)]
                           + [("tail", 5)] + [("new_tail", i) for i in range(1, 6)])
        need([(o["phase"], o["step"]) for o in row["observations"]] == expected_frames, "frame roster")
        for obs in row["observations"]:
            path = RUN / obs["file"]
            need(path.parent == RUN and not path.is_symlink()
                 and physical.previous.sha(path) == obs["file_sha256"], "frame file")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == obs["pixel_sha256"], "frame pixels")
            frames += 1
        cp10 = [o for o in row["observations"] if o["phase"] == "tail" and o["step"] == 5]
        need(len(cp10) == 1 and all(cp10[0][k] == role["cp10_checkpoint"][k]
             for k in ("pixel_sha256", "state", "snapshot")), "cp10 identity")
        need(len(prior["transitions"]) == 25
             and all(np.array_equal(a["action"], b["action"]) and a["state"] == b["state"]
                     for a, b in zip(row["transitions"][:20], prior["transitions"][:20], strict=True)),
             "paired replay through cp10")
        before, new, previous = (measures(x, contract["goal_state"]) for x in
                                 (row["transitions"][19]["state"], row["transitions"][-1]["state"],
                                  prior["transitions"][-1]["state"]))
        rows.append({"arm": role["arm"], "cp10": before, "new_terminal": new, "old_terminal": previous,
                     "new_minus_old": {k: new[k] - previous[k] for k in new},
                     "new_reward": row["transitions"][-1]["reward"],
                     "old_reward": prior["transitions"][-1]["reward"],
                     "new_success": False, "old_success": False})
    return {"schema": "p1_second_feedback_saved.v1", "rows": rows,
            "frames_verified": frames, "source_report_sha256": REPORT_SHA,
            "classification": "known_development_root_descriptive",
            "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(physical.previous.write_new(args.output, analyze()))


if __name__ == "__main__":
    main()
