"""Saved-array diagnosis of actual block response, not contact detection or P1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_pusht_d2p_full_metric_bank import MODES, summarize, verify_stages
from experiments.spectral_reachability.analyze_pusht_d2p_physical_calibration import digest


def response(branch):
    before, after = np.asarray(branch["successor_state"], float), np.asarray(branch["final_state"], float)
    if before.shape != (7,) or after.shape != (7,) or not np.isfinite([before, after]).all():
        raise ValueError("finite seven-coordinate physical states required")
    angle = float(np.arctan2(np.sin(after[4]-before[4]), np.cos(after[4]-before[4])))
    delta = float(branch["final_pose"]["pose_cost"] - branch["prefix_pose"]["pose_cost"])
    if not np.isfinite(delta):
        raise ValueError("finite physical pose-cost change required")
    return {"block_xy_displacement": float(np.linalg.norm(after[2:4]-before[2:4])),
            "block_angle_displacement_abs": abs(angle),
            "block_pose_exactly_unchanged": bool(np.array_equal(before[2:5], after[2:5])),
            "agent_xy_displacement": float(np.linalg.norm(after[:2]-before[:2])),
            "suffix_pose_cost_change": delta}


def analyze(report):
    # Validate full paired bank and original causal seals before interpreting it.
    summarize(report)
    rows = []
    for row in sorted(report["rows"], key=lambda r: (r["root_id"], r["proposal_index"])):
        for mode in MODES:
            branch = row["branches"][mode]
            entry = response(branch)
            goal = np.asarray(row["goal_vector"], float)
            start = np.asarray(branch["observed_prefix_vector"], float)
            predicted = np.asarray(branch["suffix_receipt"]["predicted_terminal_vector"], float)
            actual = np.asarray(branch["final_vector"], float)
            base = float(np.mean((start-goal)**2))
            rows.append({"root_id": row["root_id"], "proposal_index": row["proposal_index"],
                         "mode": mode, **entry,
                         "predicted_rgb_cost_change": float(np.mean((predicted-goal)**2))-base,
                         "actual_rgb_cost_change": float(np.mean((actual-goal)**2))-base})
    aggregate = {}
    for mode in MODES:
        group = [r for r in rows if r["mode"] == mode]
        aggregate[mode] = {"branches": len(group),
            "exactly_no_block_pose_change": sum(r["block_pose_exactly_unchanged"] for r in group),
            "agent_moves_without_block_pose_change": sum(r["block_pose_exactly_unchanged"] and r["agent_xy_displacement"] > 0 for r in group),
            "predicted_rgb_improves_without_block_pose_change": sum(r["block_pose_exactly_unchanged"] and r["predicted_rgb_cost_change"] < 0 for r in group),
            "actual_rgb_improves_without_block_pose_change": sum(r["block_pose_exactly_unchanged"] and r["actual_rgb_cost_change"] < 0 for r in group),
            "suffix_pose_improves": sum(r["suffix_pose_cost_change"] < 0 for r in group),
            "suffix_pose_worsens": sum(r["suffix_pose_cost_change"] > 0 for r in group)}
    return {"schema": "pusht_d2p_suffix_object_response.v1", "rows": rows, "aggregate": aggregate,
            "P1_admission": False, "outcome_opened": False, "selection_changed": False,
            "scope": "Known full calibration bank, unequal solver budgets. Exact unchanged block endpoints do not prove absence of contact or intermediate motion. Diagnostic never selects actions or removes roots."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable response diagnostic required")
    report = json.loads(args.report.read_text())
    verify_stages(args.report, report)
    result = analyze(report)
    result["report_sha256"] = digest(args.report)
    result["implementation_sha256"] = {str(Path(fn.__code__.co_filename).resolve()): digest(Path(fn.__code__.co_filename))
                                        for fn in (response, summarize, verify_stages, digest)}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
