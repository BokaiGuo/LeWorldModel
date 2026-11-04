"""Verify all saved actions/images and decompose task outcomes, post hoc only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_dev303_residual_spectral_physical_20261102_main as physical

base, require, ROOT = physical.base, physical.require, physical.ROOT
CONTRACT = ROOT / "results/p1_dev303_residual_spectral_physical_contract_20261102_main_run01.json"
CONTRACT_SHA = "30e6ecf671955350445be0b61487fd5ab37440fbb222fa9559f69921347fd8d6"
RUN = ROOT / "results/p1_dev303_residual_spectral_physical_20261102_main_run01"
REPORT_SHA = "e4de331a64adec269daf1e6efd75f940068800520a66a3870f3e0d389f8b9283"


def measures(state, goal):
    state, goal = np.asarray(state, float), np.asarray(goal, float)
    require(state.shape == goal.shape == (7,) and np.isfinite(state).all()
            and np.isfinite(goal).all(), "finite native state/goal")
    angle = abs(float(np.arctan2(np.sin(state[4] - goal[4]), np.cos(state[4] - goal[4]))))
    return {"object_position_error": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "object_angle_error": angle,
            "pusher_position_error": float(np.linalg.norm(state[:2] - goal[:2])),
            "velocity_squared_error": float(np.sum((state[5:] - goal[5:]) ** 2))}


def analyze():
    require(base.sha(CONTRACT) == CONTRACT_SHA and base.read(CONTRACT) == physical.build(),
            "sealed contract/source binding")
    require(base.sha(RUN / "report.json") == REPORT_SHA, "sealed report binding")
    contract, report = base.read(CONTRACT), base.read(RUN / "report.json")
    require(report["status"] == "recorded_development" and not report["errors"]
            and report["prepared_sha256"] == CONTRACT_SHA, "complete physical panel")
    require(base.sha(RUN / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    require([row["arm"] for row in report["rows"]] == [s["arm"] for s in contract["specs"]],
            "complete ordered roster")
    goal = report["goal_state_evaluation_only"]
    rows, images = [], 0
    for spec, row in zip(contract["specs"], report["rows"], strict=True):
        require(base.read(RUN / (spec["arm"] + ".json")) == row, "row shard")
        require(row["status"] == "fixed_horizon_completed", "no censoring allowed in this analysis")
        expected = [("warmup", i + 1, [0., 0.]) for i in range(10)]
        expected += [("prefix", i + 1, a) for i, a in enumerate(spec["prefix"])]
        expected += [("tail", i + 1, a) for i, a in enumerate(spec["tail"])]
        require(len(row["transitions"]) == len(expected), "complete action trace")
        for transition, (phase, step, action) in zip(row["transitions"], expected, strict=True):
            require(transition["phase"] == phase and transition["step"] == step
                    and np.array_equal(transition["action"], action)
                    and not transition["terminated"] and not transition["truncated"], "action trace")
        expected_frames = [("warmup", i) for i in (0, 5, 10)]
        expected_frames += [("prefix", i) for i in range(1, 6)]
        expected_frames += [("tail", i) for i in range(1, 11)]
        require([(o["phase"], o["step"]) for o in row["observations"]] == expected_frames, "frame roster")
        for obs in row["observations"]:
            path = RUN / obs["file"]
            require(path.parent == RUN and not path.is_symlink(), "local frame path")
            require(base.sha(path) == obs["file_sha256"], "PNG SHA")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == obs["pixel_sha256"], "pixel SHA")
            images += 1
        root, prefix, end = [row["transitions"][i]["state"] for i in (9, 14, 24)]
        root_metrics, prefix_metrics, end_metrics = [measures(s, goal) for s in (root, prefix, end)]
        rows.append({"arm": spec["arm"], "label": spec["label"], "metric": spec["metric"],
                     "selected_before_physics": spec["selected_before_physics"],
                     "root": root_metrics, "prefix": prefix_metrics, "terminal": end_metrics,
                     "prefix_change": {k: prefix_metrics[k] - root_metrics[k] for k in root_metrics},
                     "terminal_reward": row["transitions"][-1]["reward"],
                     "object_displacement_after_prefix": float(np.linalg.norm(np.asarray(end)[2:4]
                                                                               - np.asarray(prefix)[2:4])),
                     "object_rotation_after_prefix": float(abs(np.arctan2(
                         np.sin(end[4] - prefix[4]), np.cos(end[4] - prefix[4])))),
                     "native_terminated": False})
    return {"schema": "dev303_residual_spectral_saved_analysis.v1", "rows": rows, "frame_references_verified": images,
            "physical_report_sha256": REPORT_SHA, "contract_sha256": CONTRACT_SHA,
            "classification": "posthoc_development_descriptive",
            "common_suffix_intervention_done": False, "cp5_feedback_replanning_done": False,
            "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze()
    digest = physical.prior.write_new(args.output, result)
    print(json.dumps({"sha256": digest, "arms": len(result["rows"]),
                      "frame_references_verified": result["frame_references_verified"]}))


if __name__ == "__main__":
    main()
