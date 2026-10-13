"""Aggregate matched PushT learned-adapter closure runs.

The selected arm is the contact-frame torque readout evaluated inside the
verified-hybrid planner.  This summary deliberately keeps the learned
proposal and simulator verification separate: the result is a privileged
planning-interface diagnostic, not a pure learned-dynamics or visual-planning
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="block_only_torque_readout")
    parser.add_argument("--baseline", default="state_action")
    return parser.parse_args()


def mean_ci(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("metric values must be non-empty and finite")
    # The planner already stores episode-level rows.  The normal approximation
    # is only a compact descriptive interval; seed-level rows remain primary.
    mean = float(values.mean())
    if values.size == 1:
        half = 0.0
    else:
        half = float(1.96 * values.std(ddof=1) / math.sqrt(values.size))
    return {
        "estimate": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": int(values.size),
    }


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    xy = float(row["block_xy_distance"])
    angle = float(row["angle_error"])
    agent = float(row["agent_xy_distance"])
    return {
        "success": float(bool(row["success"])),
        "block_pose_closure": float(xy < 20.0 and angle < np.pi / 9.0),
        "orientation_closure": float(angle < np.pi / 9.0),
        "block_xy_distance": xy,
        "angle_error": angle,
        "agent_xy_distance": agent,
    }


def main() -> None:
    args = parse_args()
    reports: list[dict[str, Any]] = []
    for input_dir in args.inputs:
        path = Path(input_dir) / "object_centric_learned_planner.json"
        report = json.loads(path.read_text())
        rows = [row for row in report["rows"] if row["model"] == args.model]
        baseline = [row for row in report["rows"] if row["model"] == args.baseline]
        if len(rows) == 0 or len(rows) != len(baseline):
            raise ValueError(f"missing or mismatched model rows in {path}")
        rows = sorted(rows, key=lambda row: int(row["episode"]))
        baseline = sorted(baseline, key=lambda row: int(row["episode"]))
        if [row["episode"] for row in rows] != [row["episode"] for row in baseline]:
            raise ValueError(f"baseline is not episode-matched in {path}")
        metrics = {key: np.asarray([row_metrics(row)[key] for row in rows]) for key in row_metrics(rows[0])}
        paired = {
            "success_difference": mean_ci(metrics["success"] - np.asarray([row_metrics(row)["success"] for row in baseline])),
            "block_xy_difference": mean_ci(metrics["block_xy_distance"] - np.asarray([float(row["block_xy_distance"]) for row in baseline])),
            "angle_difference": mean_ci(metrics["angle_error"] - np.asarray([float(row["angle_error"]) for row in baseline])),
        }
        reports.append({
            "input": str(path),
            "seed": int(report["protocol"]["seed"]),
            "n": len(rows),
            "metrics": {key: mean_ci(value) for key, value in metrics.items()},
            "paired_against_baseline": paired,
            "protocol": report["protocol"],
        })

    pooled_rows = []
    pooled_baseline = []
    for item in reports:
        report = json.loads((Path(item["input"])).read_text())
        pooled_rows.extend(sorted([row for row in report["rows"] if row["model"] == args.model], key=lambda row: (int(report["protocol"]["seed"]), int(row["episode"]))))
        pooled_baseline.extend(sorted([row for row in report["rows"] if row["model"] == args.baseline], key=lambda row: (int(report["protocol"]["seed"]), int(row["episode"]))))
    pooled = {key: np.asarray([row_metrics(row)[key] for row in pooled_rows]) for key in row_metrics(pooled_rows[0])}
    baseline_metrics = {key: np.asarray([row_metrics(row)[key] for row in pooled_baseline]) for key in row_metrics(pooled_baseline[0])}
    aggregate = {
        "metrics": {key: mean_ci(value) for key, value in pooled.items()},
        "paired_against_baseline": {
            "success_difference": mean_ci(pooled["success"] - baseline_metrics["success"]),
            "block_xy_difference": mean_ci(pooled["block_xy_distance"] - baseline_metrics["block_xy_distance"]),
            "angle_difference": mean_ci(pooled["angle_error"] - baseline_metrics["angle_error"]),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "selected_model": args.model,
            "matched_baseline": args.baseline,
            "seed_level_rows": len(reports),
            "pooled_cases": len(pooled_rows),
            "same_case_schedule_within_seed": True,
            "selection": "learned contact-frame torque proposal plus full-fan one-step simulator verification",
            "privileged_interface": True,
            "pure_learned_rollout": False,
        },
        "seed_reports": reports,
        "aggregate": aggregate,
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(value).all() for value in pooled.values())),
            "same_case_count_within_seed": True,
            "both_seeds_present": len(reports) == 2,
        },
        "claim_boundary": (
            "The contact-frame torque readout is a learned proposal that improves a privileged "
            "simulator-state planning interface under matched current-state verification. "
            "The artifact does not establish pure learned multi-step rollout, frozen-LeWM "
            "physical causality, visual online planning, or a universal PushT planner."
        ),
    }
    (output / "learned_adapter_closure_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = [
        "# Learned adapter closure aggregate",
        "",
        "| seed | strict success | block-pose closure | orientation closure | block XY | angle | agent XY |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        m = item["metrics"]
        lines.append(
            "| %d | %.0f%% | %.0f%% | %.0f%% | %.2f | %.3f | %.2f |" % (
                item["seed"], 100 * m["success"]["estimate"],
                100 * m["block_pose_closure"]["estimate"],
                100 * m["orientation_closure"]["estimate"],
                m["block_xy_distance"]["estimate"], m["angle_error"]["estimate"],
                m["agent_xy_distance"]["estimate"],
            )
        )
    m = aggregate["metrics"]
    lines += [
        "| pooled | %.0f%% | %.0f%% | %.0f%% | %.2f | %.3f | %.2f |" % (
            100 * m["success"]["estimate"], 100 * m["block_pose_closure"]["estimate"],
            100 * m["orientation_closure"]["estimate"], m["block_xy_distance"]["estimate"],
            m["angle_error"]["estimate"], m["agent_xy_distance"]["estimate"],
        ),
        "",
        "The matched baseline and episode-level paired differences are stored in the JSON artifact.",
        "The selected arm uses learned contact-frame proposals followed by full-fan one-step simulator verification.",
        "This is a privileged simulator-state planning-interface result, not a pure learned rollout or visual LeWM planning result.",
        "",
    ]
    (output / "learned_adapter_closure_aggregate.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "aggregate": aggregate, "sanity_checks": result["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
