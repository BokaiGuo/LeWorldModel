"""Saved-only dev303 post-hoc tail response analysis; no tuning or actions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.spatial import cKDTree

from . import p1_dev303_cp5_tail_physical_20261102_exclusive as run
from . import p1_dev303_cp5_tail_physical_saved_20261102_exclusive as saved
from . import p1_rgb_visibility_20261101_main as visibility

NAME = "p1_dev303_cp5_tail_response_analysis_20261102_exclusive"
ROOT = run.ROOT
PREP = ROOT / "results/p1_dev303_cp5_tail_physical_20261102_exclusive_PREP_DEV01.json"
DIRECTORY = ROOT / "results/p1_dev303_cp5_tail_physical_20261102_exclusive_RUN_DEV01"
PREP_SHA = "1ee6d170a4f00d64a861d04b42b5b6003c5f0f8da450341ae713ac4456c59c50"
REPORT_SHA = "f9f367dff0913ab85287461178d089ad0aa214af28beb6af0d0fc75491ae838a"


def geometry(pixels):
    masks = visibility.masks(pixels)
    obj, pusher = (np.argwhere(masks[k]) for k in ("object", "pusher"))
    if not len(obj) or not len(pusher):
        raise ValueError("empty object or pusher mask")
    d, _ = cKDTree(obj).query(pusher, k=1)
    centroid_gap = float(cKDTree(obj).query(pusher.mean(axis=0), k=1)[0])
    return {
        "nearest_mask_pixel_distance": float(np.min(d)),
        "pusher_centroid_to_object_pixel_distance": centroid_gap,
        "object_pixels": int(len(obj)), "pusher_pixels": int(len(pusher)),
        "object_components_4_connected": int(label(masks["object"])[1]),
        "pusher_components_4_connected": int(label(masks["pusher"])[1]),
        "object_touches_image_edge": bool(np.any((obj == 0) | (obj == 223))),
        "pusher_touches_image_edge": bool(np.any((pusher == 0) | (pusher == 223))),
    }


def analyze():
    audit = saved.verify(PREP, PREP_SHA, DIRECTORY, REPORT_SHA)
    run.require(audit["run_status"] == "recorded" and all(
        r["status"] == "fixed_horizon_completed" and r["tail_returned"] == 10
        for r in audit["branches"]), "complete two-arm saved run")
    report = run.base.read(DIRECTORY / "report.json")
    planner_prep = run.handoff.export.decode(
        run.handoff.planner.parent.read_fixed(run.handoff.PREP, run.handoff.PREP_SHA))
    roles = {x["role"]: x for x in planner_prep["roles"]}
    results = []
    for row in report["rows"]:
        arm = row["arm"]
        points = [p for p in row["observations"]
                  if (p["phase"], p["step"]) == ("prefix", 5) or p["phase"] == "tail"]
        transitions = [t for t in row["transitions"] if t["phase"] == "tail"]
        run.require([(p["phase"], p["step"]) for p in points]
                    == [("prefix", 5)] + [("tail", i) for i in range(1, 11)]
                    and [t["step"] for t in transitions] == list(range(1, 11)),
                    "exact observation and action sequence")
        direction = np.asarray(roles[arm]["rgb_contact_geometry"]["contact_direction_raw_xy"],
                               dtype=np.float32)
        direction /= np.linalg.norm(direction)
        actions = np.asarray([t["action"] for t in transitions], dtype="<f4")
        projection = actions @ direction
        base_pose = np.asarray(points[0]["state"][2:5], np.float64)
        rows = []
        for step, point in enumerate(points):
            with Image.open(DIRECTORY / point["file"]) as im:
                image = np.asarray(im.convert("RGB"), np.uint8)
            state = np.asarray(point["state"], np.float64)
            pose = state[2:5]
            delta = pose - base_pose
            rows.append({
                "tail_step": step, "image_file": point["file"],
                "pixel_sha256": point["pixel_sha256"], **geometry(image),
                "actual_action": None if step == 0 else actions[step - 1].tolist(),
                "action_l2": None if step == 0 else float(np.linalg.norm(actions[step - 1])),
                "projection_on_cp5_direction": None if step == 0 else float(projection[step - 1]),
                "object_pose_xy_angle": pose.tolist(),
                "object_xy_displacement_from_cp5": float(np.linalg.norm(delta[:2])),
                "object_angle_change_from_cp5": float(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))),
            })
        bank = actions.reshape(1, 2, 5, 2)
        value = float(run.handoff.planner.parent.engine._projection(bank, direction)[0])
        results.append({
            "arm": arm, "cp5_direction_xy": direction.tolist(),
            "first_three_projection_sum": value,
            "projection_threshold": run.handoff.planner.parent.engine.MIN_PROJECTION,
            "projection_pass": value >= run.handoff.planner.parent.engine.MIN_PROJECTION,
            "max_object_xy_displacement_from_cp5": max(x["object_xy_displacement_from_cp5"] for x in rows),
            "max_absolute_object_angle_change_from_cp5": max(abs(x["object_angle_change_from_cp5"]) for x in rows),
            "rows": rows,
        })
    return {
        "schema": NAME + ".v1", "classification": "post_hoc_development_diagnostic",
        "source_report_sha256": REPORT_SHA, "source_preparation_sha256": PREP_SHA,
        "source_saved_verification": audit, "arms": results,
        "definitions": {
            "mask_distance": "minimum Euclidean distance between centers of renderer-specific visible object and pusher mask pixels",
            "action_l2": "L2 norm of saved actual two-component action in native action coordinates",
            "pose": "saved native state indices 2,3,4; simulator coordinates and radians",
            "projection": "sum of dot products of first 3 tail actions with fixed normalized cp5 RGB direction",
            "limitations": "visible pixels may omit occluded boundaries; sampled frames do not rule out transient contact; no causal intervention",
        },
        "P1_admission": False, "independent_confirmation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    run.require(not os.path.lexists(output), "exclusive output")
    value = analyze()
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "sha256": run.handoff.digest(raw)}))


if __name__ == "__main__":
    main()
