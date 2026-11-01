"""Create planner-only inputs for all joint/zero/greedy cp5 anchors."""
import json
from pathlib import Path

import numpy as np
from . import p1_joint_h10_20261030 as h10
from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind
from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as planner

ROOT = h10.ROOT
OUT = ROOT / "results/p1_joint_cp5_bundle_20261030_RUN01.json"
PREP = ROOT / "results/p1_joint_h10_20261030_prepare_RUN01.json"
PREP_SHA = "71579e186113a71d5571006de8519633950b05c32e887bb86563db1bba946b34"


def build():
    require = h10.require
    require(h10.sha(PREP) == PREP_SHA, "H10 prepare hash")
    fixed = h10.joint.runner.native.read(PREP, cap=64*1024**2)
    require(fixed == h10.build(), "H10 prepare rebuild")
    require(planner.sha(planner.BUNDLE) == planner.BUNDLE_SHA, "old blind bundle")
    previous = json.loads(planner.BUNDLE.read_bytes())
    _, _, _, tasks = h10.joint.replay.old.sources()
    source = h10.joint.runner.native.read(h10.joint.JOIN, cap=32*1024**2)
    rows = []
    for i, anchor in enumerate(fixed["anchors"]):
        require(anchor["index"] == i, "anchor order")
        slot = 2*anchor["root_slot"] + int(anchor["condition"] == "on")
        task = tasks["requests"][slot]["task"]
        raw = bytes.fromhex(source["task_rows"][slot]["warmup_raw_float32_hex"])
        checkpoint = anchor["checkpoint"]
        if anchor["role"] == "selected":
            image_path = h10.PHYS / checkpoint["png"]
            require(image_path.parent == h10.PHYS and image_path.is_file()
                    and not image_path.is_symlink(), "selected frame")
            row = blind.make_row(anchor, task, np.frombuffer(raw, "<f4").reshape(10, 2),
                                 image_path.read_bytes())
        else:
            row = previous["rows"][i]
            blind.validate_row(row)
            require(row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
                    and row["cp5_pixel_sha256"] == checkpoint["pixel_sha256"]
                    and row["root_latent_history"] == task["root_context"]["latent_history"]
                    and row["root_action_history"] == task["root_context"]["action_history"]
                    and row["goal_latent"] == task["goal"] and row["action_mean"] == task["action_mean"]
                    and row["action_std"] == task["action_std"], "old planner input identity")
        rows.append(row)
    require(len(rows) == 24 and h10.sha(PREP) == PREP_SHA, "roster and drift")
    return {"schema": "codex_p1_three_prefix_blind_inputs_v1", "rows": rows,
            "fixed_roster": 24, "planner_file_isolation_established": False,
            "GPU_run": False, "physical_run": False, "P1_admission": False}


def main():
    require = h10.require
    require(not OUT.exists() and not OUT.is_symlink(), "exclusive bundle")
    digest = h10.run._write_new(OUT, build())
    print(json.dumps({"sha256": digest, "rows": 24, "physical_authorized": False}))


if __name__ == "__main__":
    main()
