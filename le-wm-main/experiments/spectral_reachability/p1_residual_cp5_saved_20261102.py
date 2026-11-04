"""Saved-only verification of the complete residual-prefix feedback panel."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_residual_cp5_physical_20261102 as physical
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

ROOT = physical.ROOT
CONTRACT = ROOT / "results/p1_residual_cp5_physical_20261102_PREP_DEV01.json"
CONTRACT_SHA = "97d9abf41881d335b8335b8901e1300ed5ef466348a7b3a362b518e4ac927883"
RUN = ROOT / "results/p1_residual_cp5_physical_20261102_RUN_DEV01"
REPORT_SHA = "c66b42a3ab056b5a422515ae6932b6455d7dc994f6771d86861ac6af323b7191"
need = physical.require


def analyze():
    contract = physical.read(CONTRACT, CONTRACT_SHA)
    need(contract == physical.build(), "contract closure")
    report = physical.read(RUN / "report.json", REPORT_SHA)
    need(report["status"] == "recorded" and not report["errors"], "complete physical run")
    need(report["prepare_sha256"] == CONTRACT_SHA, "contract binding")
    need(physical.sha(RUN / "journal.jsonl") == report["journal_sha256"], "journal")
    need([r["arm"] for r in report["rows"]] == [r["arm"] for r in contract["roles"]], "roster")
    rows, frames = [], 0
    for role, row in zip(contract["roles"], report["rows"], strict=True):
        need(physical.read(RUN / (role["arm"] + ".row.json")) == row, "row shard")
        # This fixed-horizon analysis abstains on censored branches.
        need(row["status"] == "fixed_horizon_completed", "censored branch needs separate analysis")
        expected = [("warmup", i + 1, [0., 0.]) for i in range(10)]
        expected += [("prefix", i + 1, a) for i, a in enumerate(role["prefix"])]
        expected += [("tail", i + 1, a) for i, a in enumerate(role["tail"])]
        need(len(row["transitions"]) == len(expected), "complete transitions")
        for actual, (phase, step, action) in zip(row["transitions"], expected, strict=True):
            need(actual["phase"] == phase and actual["step"] == step
                 and np.array_equal(actual["action"], action)
                 and not actual["terminated"] and not actual["truncated"], "action sequence")
        expected_frames = [("warmup", i) for i in (0, 5, 10)]
        expected_frames += [("prefix", i) for i in range(1, 6)]
        expected_frames += [("tail", i) for i in range(1, 11)]
        need([(o["phase"], o["step"]) for o in row["observations"]] == expected_frames, "frame roster")
        for obs in row["observations"]:
            path = RUN / obs["file"]
            need(path.parent == RUN and not path.is_symlink(), "local image path")
            need(physical.sha(path) == obs["file_sha256"], "image file SHA")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == obs["pixel_sha256"], "pixel SHA")
            frames += 1
        prefix_frames = [o for o in row["observations"] if o["phase"] == "prefix"]
        for actual, original in zip(prefix_frames, role["prefix_checkpoints"], strict=True):
            need(all(actual[k] == original[k] for k in
                     ("phase", "step", "pixel_sha256", "state", "snapshot")), "full prefix replay")
        states = [row["transitions"][i]["state"] for i in (9, 14, 24)]
        values = [measures(s, contract["goal_state"]) for s in states]
        rows.append({
            "arm": role["arm"], "label": role["label"], "metric": role["metric"],
            "root": values[0], "prefix": values[1], "terminal": values[2],
            "prefix_change": {k: values[1][k] - values[0][k] for k in values[0]},
            "tail_change": {k: values[2][k] - values[1][k] for k in values[0]},
            "terminal_reward": row["transitions"][-1]["reward"],
            "terminated": False, "truncated": False,
        })
    return {"schema": "p1_residual_cp5_saved.v1", "rows": rows,
            "frame_references_verified": frames, "report_sha256": REPORT_SHA,
            "contract_sha256": CONTRACT_SHA, "P1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False,
            "classification": "known_development_root_descriptive"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    print(physical.write_new(args.output, result))
