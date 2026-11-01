"""Exploratory full-denominator summaries from verified saved physical rows."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from . import codex_p1_cp5_physical_saved_20261030_v1 as saved


def analyze(receipt):
    need = saved.need
    need(receipt["status"] == "saved_checks_passed" and receipt["fixed_rows"] == len(receipt["rows"]) == 24,
         "verified all-24 receipt")
    anchors = saved.run.h10.native.read(saved.run.h10.INPUT, cap=64*1024**2)["anchors"]
    out = []
    for i, (row, anchor) in enumerate(zip(receipt["rows"], anchors, strict=True)):
        need(row["index"] == anchor["index"] == i and row["role"] == anchor["role"]
             and row["root_slot"] == anchor["root_slot"] and row["condition"] == anchor["condition"],
             "identity")
        state = row["completed_endpoint"]
        fields = None
        if state is not None:
            state, goal = np.asarray(state, np.float64), np.asarray(anchor["goal"], np.float64)
            need(state.shape == goal.shape == (7,) and np.isfinite(state).all() and np.isfinite(goal).all(),
                 "finite state and goal")
            angle = state[4]-goal[4]
            fields = {"native_success": row["last_accepted_state_metrics"]["native_success"],
                      "object_position_error": float(np.linalg.norm(state[2:4]-goal[2:4])),
                      "angle_error_radians": float(abs(math.atan2(math.sin(angle), math.cos(angle)))),
                      "pusher_position_error": float(np.linalg.norm(state[:2]-goal[:2]))}
        out.append({"index": i, "root_slot": row["root_slot"], "condition": row["condition"],
                    "arm": row["arm"], "role": row["role"], "status": row["status"],
                    "returned_suffix": row["returned_suffix"], "endpoint_metrics": fields,
                    "actual_resources": row["counts"]})
    return {"schema": "codex_p1_cp5_physical_analysis_20261030_v1", "rows": out,
            "fixed_denominator": 24, "known_roots": True, "P1_admission": False,
            "independent_confirmation": False, "os_isolation_certified": False,
            "selection_on_outcome": False, "fairness_proven": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    need = saved.need
    need(saved.run.sha(a.receipt) == a.sha256, "receipt SHA")
    result = analyze(json.loads(Path(a.receipt).read_bytes()))
    path = Path(a.output)
    need(path.is_absolute() and path.parent == saved.run.ROOT/"results" and
         path.name.startswith("codex_p1_cp5_physical_analysis_20261030_v1_"), "exclusive output")
    print(json.dumps({"sha256": saved.run.h10._write_new(path, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
