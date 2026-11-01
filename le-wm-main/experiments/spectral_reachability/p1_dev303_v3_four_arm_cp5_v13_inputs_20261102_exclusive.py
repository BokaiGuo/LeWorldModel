"""Four-arm saved cp5 input contract for later V13 planning, no model or physics."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_four_arm_physical_20261102_exclusive as physical
from . import p1_dev303_v3_four_arm_physical_saved_20261102_exclusive as saved
from . import p1_dev303_cp5_rgb_action_export_20261102_exclusive as rgb
from . import p1_dev303_cp5_equal_v13_replan_20261102_exclusive as v13
from . import p1_dev303_cp5_dual_encode_20261102_exclusive as encode
from . import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as planner

ROOT, base, require = physical.ROOT, physical.base, physical.require
NAME = "p1_dev303_v3_four_arm_cp5_v13_inputs_20261102_exclusive"
PREP = ROOT / "results/p1_dev303_v3_four_arm_physical_20261102_exclusive_PREP_02.json"
PREP_SHA = "8e2692bf54cff2ddf1a7abcb4d447861ef25cc31836f64770f230d1358bed3d7"
RUN = ROOT / "results/p1_dev303_v3_four_arm_physical_20261102_exclusive_RUN_DEV01"
REPORT_SHA = "c74363f2c5a0f34866e9becdc0b9891f83c48c67872f88e69d14fff94c13ed93"
ROLES = ("selected", "zero", "task_greedy_direct", "random")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    verification = saved.verify(PREP, PREP_SHA, RUN, REPORT_SHA)
    require(verification["run_status"] == "recorded_development"
            and [row["arm"] for row in verification["branches"]] == list(ROLES)
            and all(row["status"] == "fixed_horizon_completed" and row["prefix_returned"] == 5
                    for row in verification["branches"]), "all four verified actual prefixes")
    prefix_contract, report = base.read(PREP), base.read(RUN / "report.json")
    arrays, receipt, _ = physical.prior.source_data()
    task_source = v13.task_source
    task_raw = v13.read_fixed(task_source.TASK, task_source.TASK_SHA)
    envelope = rgb.decode(gzip.decompress(task_raw))
    task_prep = rgb.decode(v13.read_fixed(v13.TASK_PREP, v13.TASK_PREP_SHA))
    task = task_prep["task"]
    require(envelope["task"] == task and envelope["development_seed"] == physical.prior.SEED
            and envelope["P1_admission"] is False
            and envelope["new_independent_confirmation"] is False
            and envelope["blind_source_sha256"] == task_source.BLIND_SHA
            and task_prep["source_receipt_sha256"] == task_source.RECEIPT_SHA
            and task_prep["pixel_sha256"] == [digest(arrays["rgb_history"][-1].tobytes()),
                                              digest(arrays["goal_rgb"].tobytes())]
            and task_prep["model_sha256"] == v13.engine.base.MODEL_SHA256
            and task_prep["P1_admission"] is False, "task, goal and model binding")
    mean, std = np.asarray(task["action_mean"], float), np.asarray(task["action_std"], float)
    history = np.asarray(task["root_context"]["latent_history"], float)
    packet = np.asarray(task["root_context"]["action_history"][-1], float)
    goal = np.asarray(task["goal"], float)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all()
            and task["root_raw_action_history"] == np.zeros((10, 2), np.float32).tolist()
            and packet.shape == (10,) and np.allclose(packet, np.tile(-mean / std, 5), atol=1e-6, rtol=0)
            and history.ndim == 2 and history.shape[0] >= 2 and history.shape[1] == task["latent_dim"]
            and goal.shape == (task["latent_dim"],)
            and np.isfinite(history).all() and np.isfinite(goal).all(), "task normalization and context")
    roles = []
    for spec, row in zip(prefix_contract["arms"], report["rows"], strict=True):
        arm = row["arm"]
        require(arm == spec["role"], "role order")
        checkpoints = [p for p in row["observations"] if p["phase"] == "prefix"]
        require([p["step"] for p in checkpoints] == list(range(1, 6)), "five frames")
        cp = checkpoints[-1]
        pixels = rgb._frame(RUN, cp)
        actions = [t["action"] for t in row["transitions"] if t["phase"] == "prefix"]
        array = np.asarray(actions, dtype="<f4")
        require(array.shape == (5, 2) and array.tolist() == spec["actions"]
                and digest(array.tobytes()) == spec["actions_sha256"], "actual prefix identity")
        geometry = v13.engine._geometry(pixels)
        roles.append({"role": arm, "cp5_png": str((RUN / cp["file"]).resolve()),
                      "cp5_file_sha256": cp["file_sha256"], "cp5_pixel_sha256": cp["pixel_sha256"],
                      "actual_prefix": array.tolist(), "actual_prefix_sha256": digest(array.tobytes()),
                      "prefix_checkpoints": checkpoints, "rgb_contact_geometry": geometry})
    paths = [Path(__file__), Path(saved.__file__), Path(physical.__file__), Path(rgb.__file__),
             Path(v13.__file__), Path(v13.engine.__file__), Path(v13.engine.base.__file__),
             Path(v13.engine.base.common.__file__), Path(v13.engine.base.p0.__file__),
             Path(v13.engine.contact.__file__), Path(encode.__file__), Path(planner.__file__),
             v13.TASK_PREP, task_source.TASK, PREP, RUN / "report.json"]
    paths += [Path(row["cp5_png"]) for row in roles]
    return {"schema": NAME + ".v1", "classification": "posthoc_development_not_independent",
            "roles": roles, "task": task, "task_sha256": task_source.TASK_SHA,
            "policy": task_prep["policy"], "model_sha256": task_prep["model_sha256"],
            "goal_rgb_sha256": digest(arrays["goal_rgb"].tobytes()),
            "source_report_sha256": REPORT_SHA, "source_preparation_sha256": PREP_SHA,
            "random_seed": v13.SEED, "initial_mean": np.zeros((2, 5, 2), np.float32).tolist(),
            "budget_target_per_role": {"rgb_encode_calls": 1, "action_encoder_calls": 10,
                                       "predict_calls": 10, "predict_samples": 3200},
            "ledger_verified": False, "equal_full_budget_certified": False,
            "physical_actions_authorized": False, "P1_admission": False,
            "independent_confirmation": False, "bindings": {str(p.resolve()): base.sha(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build()
    sha = physical.prior.write_new(args.output, value)
    print(json.dumps({"path": str(args.output), "sha256": sha}))


if __name__ == "__main__":
    main()
