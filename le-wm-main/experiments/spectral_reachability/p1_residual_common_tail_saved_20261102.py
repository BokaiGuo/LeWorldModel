"""Verify the fixed development intervention without selecting outcomes."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_residual_common_tail_20261102 as common

ROOT, base, need = common.ROOT, common.base, common.require
CONTRACT = ROOT / "results/p1_residual_common_tail_contract_20261102_run01.json"
CONTRACT_SHA = "8fb3a5215a0e427f86f4c9d96466cd74b42626215443e07e46601d776fc0384f"
RUN = ROOT / "results/p1_residual_common_tail_20261102_run01"
REPORT_SHA = "1794d41d1f107964b4e033d0a164ab705c4de2f1e3e1b00a80046e8e8ac5d9d8"


def verify():
    need(base.sha(CONTRACT) == CONTRACT_SHA, "contract hash")
    contract = base.read(CONTRACT)
    need(contract == common.build(), "source and action closure")
    need(base.sha(RUN / "report.json") == REPORT_SHA, "report hash")
    report = base.read(RUN / "report.json")
    need(report["status"] == "recorded_development" and not report["errors"], "complete run")
    need(report["contract_sha256"] == CONTRACT_SHA, "contract binding")
    need(base.sha(RUN / "journal.jsonl") == report["journal_sha256"], "journal hash")
    need([r["arm"] for r in report["rows"]] == [r["arm"] for r in contract["specs"]], "roster")
    frames = 0
    for spec, row in zip(contract["specs"], report["rows"], strict=True):
        need(base.read(RUN / (row["arm"] + ".json")) == row, "row shard")
        need(row["status"] == "fixed_horizon_completed", "censored or failed branch")
        expected = [("warmup", i + 1, [0., 0.]) for i in range(10)]
        expected += [("prefix", i + 1, a) for i, a in enumerate(spec["prefix"])]
        expected += [("tail", i + 1, a) for i, a in enumerate(spec["tail"])]
        need(len(row["transitions"]) == len(expected), "transition count")
        for actual, (phase, step, action) in zip(row["transitions"], expected, strict=True):
            need(actual["phase"] == phase and actual["step"] == step
                 and np.array_equal(actual["action"], action)
                 and not actual["terminated"] and not actual["truncated"], "action trace")
        frame_roster = [("warmup", i) for i in (0, 5, 10)]
        frame_roster += [("prefix", i) for i in range(1, 6)]
        frame_roster += [("tail", i) for i in range(1, 11)]
        need([(o["phase"], o["step"]) for o in row["observations"]] == frame_roster, "frame roster")
        for obs in row["observations"]:
            path = RUN / obs["file"]
            need(path.parent == RUN and not path.is_symlink(), "frame location")
            need(base.sha(path) == obs["file_sha256"], "file hash")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == obs["pixel_sha256"], "pixel hash")
            frames += 1
    return {"schema": "p1_residual_common_tail_saved.v1", "report_sha256": REPORT_SHA,
            "contract_sha256": CONTRACT_SHA, "branches_verified": len(report["rows"]),
            "frame_references_verified": frames, "P1_admission": False,
            "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(common.physical.prior.write_new(args.output, verify()))
