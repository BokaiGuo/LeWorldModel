"""Compare true-image RGB position geometry to sealed physical endpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.a0_pointmaze_failure_real_rgb_metric_20261103 import (
    ORACLE, ORACLE_SHA, OUTPUT as RGB, PLANS, PLANS_SHA,
)
from experiments.a0_pointmaze_rgb_geometry_state_20261103 import CAMERA, CAMERA_SHA, estimate_state
from experiments.a0_pointmaze_rgb_initial_candidate_oracle_20261103 import pairwise_ranking
from experiments.a1_pointmaze_causal_extra_cem_budget_20261103 import OLD, OLD_SHA
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

RGB_SHA = "714caf46a32abc659934870c9edfac8b442e53f95718e5fa8b29663bf2c2afc3"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_FAILURE_RGB_GEOMETRY_ALIGNMENT_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/a0_pointmaze_failure_rgb_geometry_alignment_20261103_DEV01.json"
ARRAYS = OUTPUT.with_suffix(".npz")


def project(image, camera, offset):
    try:
        state, _ = estimate_state(image, camera, offset)
        return state[:2].astype(np.float64), None
    except (ValueError, RuntimeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main():
    if OUTPUT.exists() or ARRAYS.exists():
        raise FileExistsError("refusing overwrite")
    sources = ((RGB, RGB_SHA), (ORACLE, ORACLE_SHA), (PLANS, PLANS_SHA),
               (CAMERA, CAMERA_SHA), (OLD, OLD_SHA))
    for path, expected in sources:
        if sha256_file(path) != expected:
            raise RuntimeError(f"source changed: {path}")
    rgb, oracle, plans, camera, old = (json.loads(p.read_text()) for p, _ in sources)
    for path, expected in ((RGB.with_suffix(".npz"), rgb["arrays_sha256"]),
                           (ORACLE.with_suffix(".npz"), oracle["arrays_sha256"]),
                           (PLANS.with_suffix(".npz"), plans["arrays_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source arrays changed: {path}")
    plan_rows = {row["root"]: row for row in plans["rows"]}
    arrays, rows, offset_i, started = {}, [], 0, time.monotonic()
    print("Project all sealed terminal RGBs and goal RGBs; no outcome-selected images.", flush=True)
    with np.load(RGB.with_suffix(".npz"), allow_pickle=False) as rendered, \
            np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as bank, \
            np.load(PLANS.with_suffix(".npz"), allow_pickle=False) as inputs:
        for row in rgb["rows"]:
            root, method, phase, n = row["root"], row["method"], row["phase"], row["n"]
            key = f"r{root}_{method}_{phase}"
            goal, goal_error = project(inputs[f"r{root}_goal_rgb"], camera["metadata"],
                                       old["body_offset_from_static_scene"])
            positions = np.full((n, 2), np.nan, np.float64)
            errors = []
            for j, image in enumerate(rendered["rgb10"][offset_i:offset_i+n]):
                position, error = project(image, camera["metadata"], old["body_offset_from_static_scene"])
                if error is not None:
                    errors.append({"index": j, "error": error})
                else:
                    positions[j] = position
            offset_i += n
            arrays[f"{key}_rgb_positions"] = positions
            physical_pos = bank[f"{key}_endpoints"][:, :2]
            physical_goal = np.asarray(plan_rows[root]["goal_position_eval_only"], np.float64)
            summary = {"root": root, "method": method, "phase": phase, "n": n,
                       "goal_detection_error": goal_error, "image_detection_errors": errors,
                       "physical_goal_eval_only": physical_goal.tolist()}
            if goal is not None and not errors:
                image_distance = np.linalg.norm(positions-goal, axis=1)
                physical_distance = bank[f"{key}_distances"]
                score = rendered[f"{key}_real_cost"]
                arrays[f"{key}_rgb_distances"] = image_distance
                choice = int(np.argmin(image_distance))
                summary.update({"rgb_goal": goal.tolist(),
                                "goal_position_error": float(np.linalg.norm(goal-physical_goal)),
                                "terminal_position_error_mean": float(np.linalg.norm(positions-physical_pos, axis=1).mean()),
                                "terminal_position_error_max": float(np.linalg.norm(positions-physical_pos, axis=1).max()),
                                "rgb_distance_vs_physical": pairwise_ranking(image_distance, physical_distance),
                                "latent_vs_rgb_distance": pairwise_ranking(score, image_distance),
                                "rgb_oracle_choice": choice,
                                "rgb_oracle_physical_distance": float(physical_distance[choice])})
            rows.append(summary)
            print(f"{key}: {len(errors)} terminal marker failures, goal error={goal_error}", flush=True)
    if offset_i != 624:
        raise RuntimeError("candidate count mismatch")
    with ARRAYS.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {"schema": "a0_pointmaze_failure_rgb_geometry_alignment.v1", "rgb_sha256": RGB_SHA,
              "oracle_sha256": ORACLE_SHA, "plans_sha256": PLANS_SHA, "camera_sha256": CAMERA_SHA,
              "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
              "arrays_sha256": sha256_file(ARRAYS), "image_count": offset_i,
              "wall_seconds": time.monotonic()-started,
              "scope": "real-future RGB oracle; outcome-selected exposed roots; no P1", "rows": rows}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
