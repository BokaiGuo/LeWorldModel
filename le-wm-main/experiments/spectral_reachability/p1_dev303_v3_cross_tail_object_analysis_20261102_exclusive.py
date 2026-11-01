"""Recomputable saved-only object-pose analysis; no physics or model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import p1_dev303_v3_cross_common_tail_20261102_exclusive as source
from . import p1_dev303_v3_cross_common_tail_saved_20261102_exclusive as saved

ROOT, base, require = source.ROOT, source.base, source.require
NAME = "p1_dev303_v3_cross_tail_object_analysis_20261102_exclusive"
PREP = ROOT / "results/p1_dev303_v3_cross_common_tail_20261102_exclusive_PREP_01.json"
PREP_SHA = "b93d5490192830c9f31abde30d7eb2b7577d89006da056e3b48f4755a1ebe88a"
DIRECTORY = ROOT / "results/p1_dev303_v3_cross_common_tail_20261102_exclusive_RUN_DEV01"
REPORT_SHA = "51f783344256f5edad4523602621f759c0fac1839c971dedd86abcdceb67471a"


def object_metrics(state, goal):
    require(len(state) >= 5 and len(goal) >= 5
            and all(math.isfinite(float(x)) for x in list(state[:5]) + list(goal[:5])),
            "finite pose")
    x, y, angle = map(float, state[2:5])
    gx, gy, ga = map(float, goal[2:5])
    delta = math.atan2(math.sin(angle - ga), math.cos(angle - ga))
    return {"xy": [x, y], "angle_rad": angle,
            "goal_xy_error": math.hypot(x - gx, y - gy),
            "goal_angle_error_rad": abs(delta), "goal_signed_angle_error_rad": delta}


def summarize(branches, rows, goal):
    require(len(branches) == len(rows), "branch roster")
    result = []
    for spec, row in zip(branches, rows, strict=True):
        require(row["arm"] == spec["arm"], "branch identity")
        cp = [p for p in row["observations"] if p["phase"] == "prefix" and p["step"] == 5]
        require(len(cp) <= 1, "cp5 uniqueness")
        tail = [t for t in row["transitions"] if t["phase"] == "tail"]
        require([t["step"] for t in tail] == list(range(1, len(tail) + 1))
                and len(tail) <= 10, "tail steps")
        # Only an observed return and matching saved frame can establish step 10.
        end = [p for p in row["observations"] if p["phase"] == "tail" and p["step"] == 10]
        require(len(end) <= 1, "terminal frame uniqueness")
        complete = bool(len(tail) == 10 and len(end) == 1 and
                        end[0]["state"] == tail[-1]["state"] and row["status"] != "failed")
        native = (True if any(t["terminated"] for t in tail) else
                  False if row["status"] in ("fixed_horizon_completed", "native_early_stop")
                  and tail else None)
        result.append({"arm": spec["arm"], "prefix_role": spec["prefix_role"],
                       "tail_role": spec["tail_role"], "status": row["status"],
                       "cp5_object": object_metrics(cp[0]["state"], goal) if cp else None,
                       "step10_object": object_metrics(tail[-1]["state"], goal) if complete else None,
                       "last_observed_object": object_metrics(tail[-1]["state"], goal) if tail else None,
                       "tail_steps_returned": len(tail), "horizon10_observed": complete,
                       "censored_horizon10": not complete, "native_success_observed": native,
                       "terminated": tail[-1]["terminated"] if tail else None,
                       "truncated": tail[-1]["truncated"] if tail else None,
                       "relative_to_zero": None})
    for row in result:
        controls = [r for r in result if r["prefix_role"] == row["prefix_role"] and r["tail_role"] == "zero"]
        require(len(controls) == 1, "one zero tail per prefix")
        zero = controls[0]["step10_object"]
        current = row["step10_object"]
        if zero is not None and current is not None:
            row["relative_to_zero"] = {
                "goal_xy_error_difference": current["goal_xy_error"] - zero["goal_xy_error"],
                "goal_angle_error_rad_difference": current["goal_angle_error_rad"] - zero["goal_angle_error_rad"],
            }
    return result


def analyze():
    verified = saved.verify(PREP, PREP_SHA, DIRECTORY, REPORT_SHA)
    _, receipt, _ = source.prior.source_data()
    report = base.read(DIRECTORY / "report.json")
    goal = receipt["history"]["goal"]
    rows = summarize(base.read(PREP)["branches"], report["rows"], goal)
    return {
        "schema": NAME + ".v1", "classification": "posthoc_development_only",
        "prepare_sha256": PREP_SHA, "source_report_sha256": REPORT_SHA,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "saved_verifier_status": verified["status"], "goal_object_xy_angle": goal[2:5],
        "definitions": {
            "position": "Euclidean distance of state[2:4] from goal[2:4]",
            "angle": "absolute atan2(sin(state[4]-goal[4]),cos(state[4]-goal[4])) in radians, 2pi periodic",
            "relative_to_zero": "branch step10 error minus zero-tail step10 error for same prefix; negative means smaller",
            "native_success": "observed tail terminated flag; native eval_state includes pusher and object position",
            "censoring": "step10 return and matching frame required; no padding, last observed separate",
            "reward": "not used; native reward is whole-state distance",
        },
        "rows": rows, "P1_admission": False, "independent_confirmation": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = analyze()
    digest = source.prior.write_new(args.output, value)
    print(json.dumps({"output": str(args.output), "sha256": digest, "rows": len(value["rows"])}))


if __name__ == "__main__":
    main()
