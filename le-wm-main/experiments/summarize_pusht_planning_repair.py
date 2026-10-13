"""Paired statistics for the validated PushT planning-objective repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed8421", type=Path, required=True)
    p.add_argument("--seed9173", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261006)
    return p.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = json.loads(path.read_text())
    return report["reports"]["joint_endpoint"], report["reports"]["receding_block_pose"]


def paired_bootstrap(delta: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(delta)),
    }


def finite_summary(values: list[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def main() -> None:
    args = parse_args()
    pairs = []
    for source in (args.seed8421, args.seed9173):
        joint, repaired = load_rows(source)
        if len(joint) != len(repaired):
            raise ValueError(f"unmatched rows in {source}")
        pairs.extend(zip(joint, repaired))
    if len(pairs) != 16:
        raise ValueError(f"expected the two matched 8-case runs, got {len(pairs)}")

    metrics = {
        "success": lambda row: float(bool(row["success"])),
        "block_pose_success": lambda row: float(bool(row["block_pose_success"])),
        "pre_return_block_xy_distance": lambda row: float(row["pre_return_block_xy_distance"]),
        "final_block_xy_distance": lambda row: float(row["final_block_xy_distance"]),
        "pre_return_angle_error": lambda row: float(row["pre_return_angle_error"]),
        "final_angle_error": lambda row: float(row["final_angle_error"]),
        "final_agent_xy_distance": lambda row: float(row["final_agent_xy_distance"]),
    }
    comparisons = {}
    for name, getter in metrics.items():
        baseline = np.asarray([getter(pair[0]) for pair in pairs], dtype=np.float64)
        repaired = np.asarray([getter(pair[1]) for pair in pairs], dtype=np.float64)
        delta = repaired - baseline
        comparisons[name] = {
            "baseline": finite_summary(baseline.tolist()),
            "receding_repair": finite_summary(repaired.tolist()),
            "delta_receding_minus_baseline": paired_bootstrap(delta, args.bootstrap_draws, args.seed + len(comparisons)),
        }

    baseline_success = np.asarray([bool(pair[0]["success"]) for pair in pairs])
    repaired_success = np.asarray([bool(pair[1]["success"]) for pair in pairs])
    baseline_pose = np.asarray([bool(pair[0]["block_pose_success"]) for pair in pairs])
    repaired_pose = np.asarray([bool(pair[1]["block_pose_success"]) for pair in pairs])
    report = {
        "protocol": {
            "sources": [str(args.seed8421), str(args.seed9173)],
            "n": len(pairs),
            "pairing": "case index within each matched seed run; the two arms reused each simulator initial state",
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "comparisons": comparisons,
        "paired_contingency": {
            "task_success": {
                "both_fail": int(np.sum(~baseline_success & ~repaired_success)),
                "baseline_only": int(np.sum(baseline_success & ~repaired_success)),
                "repair_only": int(np.sum(~baseline_success & repaired_success)),
                "both_success": int(np.sum(baseline_success & repaired_success)),
            },
            "block_pose_success": {
                "both_fail": int(np.sum(~baseline_pose & ~repaired_pose)),
                "baseline_only": int(np.sum(baseline_pose & ~repaired_pose)),
                "repair_only": int(np.sum(~baseline_pose & repaired_pose)),
                "both_success": int(np.sum(baseline_pose & repaired_pose)),
            },
        },
        "claim_boundary": (
            "Privileged simulator-state matched planning diagnosis. The paired comparison tests objective decoupling, "
            "receding replanning, and delayed safe return. It is not evidence for frozen-LeWM visual online planning, "
            "a learned physical adapter, universal causal action effects, or general PushT performance."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning_repair_paired_statistics.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT planning repair: paired statistics",
        "",
        "The two arms use the same simulator initial states within each seed run; all intervals are paired bootstrap 95% CIs.",
        "",
        "| metric | joint endpoint mean | receding repair mean | repair minus baseline (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for name, item in comparisons.items():
        ci = item["delta_receding_minus_baseline"]
        lines.append(
            "| %s | %.4f | %.4f | %.4f [%.4f, %.4f] |" % (
                name, item["baseline"]["mean"], item["receding_repair"]["mean"],
                ci["estimate"], ci["ci95_low"], ci["ci95_high"],
            )
        )
    lines += [
        "",
        "| paired task-success outcome | count |",
        "|---|---:|",
    ]
    for name, item in report["paired_contingency"].items():
        lines.extend([
            f"| {name}: both fail | {item['both_fail']} |",
            f"| {name}: baseline only | {item['baseline_only']} |",
            f"| {name}: repair only | {item['repair_only']} |",
            f"| {name}: both succeed | {item['both_success']} |",
        ])
    lines += ["", report["claim_boundary"], ""]
    (args.output / "planning_repair_paired_statistics.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "comparisons": comparisons, "paired_contingency": report["paired_contingency"]}, indent=2))


if __name__ == "__main__":
    main()
