"""Create planner-only inputs for all uphill H10 anchors."""
import json
from pathlib import Path

import numpy as np
from . import p1_uphill_h10_20261030 as h10
from . import p1_joint_cp5_bundle_20261030 as old

ROOT = h10.ROOT
PREP = ROOT / "results/p1_uphill_h10_20261030_prepare_RUN01.json"
OUT = ROOT / "results/p1_uphill_cp5_bundle_20261030_RUN01.json"
blind, planner = old.blind, old.planner


def build(prepare_sha):
    require = h10.require
    require(h10.sha(PREP) == prepare_sha, "prepare hash")
    fixed = h10.uphill.runner.native.read(PREP, cap=64 * 1024**2)
    require(fixed == h10.build(), "prepare/source rebuild")
    require(planner.sha(planner.BUNDLE) == planner.BUNDLE_SHA, "baseline bundle hash")
    previous = json.loads(planner.BUNDLE.read_bytes())
    _, _, _, tasks = h10.previous.joint.replay.old.sources()
    source = h10.previous.joint.runner.native.read(h10.previous.joint.JOIN, cap=32 * 1024**2)
    rows = []
    for i, anchor in enumerate(fixed["anchors"]):
        require(anchor["index"] == i, "anchor order")
        slot = 2 * anchor["root_slot"] + int(anchor["condition"] == "on")
        task = tasks["requests"][slot]["task"]
        raw = bytes.fromhex(source["task_rows"][slot]["warmup_raw_float32_hex"])
        checkpoint = anchor["checkpoint"]
        if anchor["role"] == "selected":
            image = h10.PHYS / checkpoint["png"]
            require(image.parent == h10.PHYS and image.is_file() and not image.is_symlink(),
                    "selected frame")
            row = blind.make_row(anchor, task, np.frombuffer(raw, "<f4").reshape(10, 2),
                                 image.read_bytes())
        else:
            row = previous["rows"][i]
            blind.validate_row(row)
            require(row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
                    and row["cp5_pixel_sha256"] == checkpoint["pixel_sha256"]
                    and row["root_latent_history"] == task["root_context"]["latent_history"]
                    and row["root_action_history"] == task["root_context"]["action_history"]
                    and row["goal_latent"] == task["goal"] and row["action_mean"] == task["action_mean"]
                    and row["action_std"] == task["action_std"], "baseline input identity")
        rows.append(row)
    require(len(rows) == 24 and h10.sha(PREP) == prepare_sha, "roster and drift")
    return {"schema": "codex_p1_three_prefix_blind_inputs_v1", "rows": rows,
            "fixed_roster": 24, "planner_file_isolation_established": False,
            "GPU_run": False, "physical_run": False, "P1_admission": False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-sha256", required=True)
    args = parser.parse_args()
    h10.require(not OUT.exists() and not OUT.is_symlink(), "exclusive output")
    digest = h10.run._write_new(OUT, build(args.prepare_sha256))
    print(json.dumps({"sha256": digest, "rows": 24, "physical_authorized": False}))


if __name__ == "__main__":
    main()
