"""Aggregate matched PushT hybrid-planner confirmation artifacts.

The hybrid arm uses the learned adapter to propose actions and a bounded
simulator successor check to commit them.  This report deliberately keeps
that task-success result separate from a pure learned-adapter claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "object_centric_learned_planner.json").read_text())
    if payload.get("claim_boundary", "").startswith("Privileged simulator-state") is False:
        raise ValueError(f"unexpected claim boundary: {path}")
    return payload


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def summarize(paths: list[Path]) -> dict[str, Any]:
    payloads = [load(path) for path in paths]
    required = {
        "planning_objective": "staged_contact_verified_hybrid_fullfan",
        "training_protocol": "contact_intervention",
        "candidate_count": 64,
        "action_block": 1,
        "episodes": 16,
        "same_cases": True,
        "same_candidate_bank_seed_schedule": True,
        "torque_feature_interface": "contact_frame",
    }
    protocol_checks = []
    for payload in payloads:
        protocol = payload["protocol"]
        protocol_checks.append({key: protocol.get(key) == value for key, value in required.items()})
    if not all(all(check.values()) for check in protocol_checks):
        raise ValueError(f"protocol mismatch: {protocol_checks}")
    if not all(finite(payload) for payload in payloads):
        raise FloatingPointError("non-finite value found in confirmation artifact")

    arms = {"state_action": [], "block_only_torque_readout": []}
    paired = []
    per_seed = []
    for payload in payloads:
        rows = payload["rows"]
        selected = {
            name: {int(row["episode"]): row for row in rows if row["model"] == name}
            for name in arms
        }
        if any(len(values) != 16 for values in selected.values()):
            raise ValueError("expected 16 rows per arm and seed")
        episodes = sorted(selected["state_action"])
        if episodes != sorted(selected["block_only_torque_readout"]):
            raise ValueError("unmatched episodes within seed")
        hybrid_rows = [selected["block_only_torque_readout"][episode] for episode in episodes]
        baseline_rows = [selected["state_action"][episode] for episode in episodes]
        per_seed.append({
            "seed": int(payload["protocol"]["seed"]),
            "n": len(episodes),
            "hybrid_success": int(sum(bool(row["strict_task_success"]) for row in hybrid_rows)),
            "baseline_success": int(sum(bool(row["strict_task_success"]) for row in baseline_rows)),
            "hybrid_success_rate": mean([{"x": bool(row["strict_task_success"])} for row in hybrid_rows], "x"),
            "baseline_success_rate": mean([{"x": bool(row["strict_task_success"])} for row in baseline_rows], "x"),
            "hybrid_block_xy": mean(hybrid_rows, "block_xy_distance"),
            "baseline_block_xy": mean(baseline_rows, "block_xy_distance"),
            "hybrid_angle": mean(hybrid_rows, "angle_error"),
            "baseline_angle": mean(baseline_rows, "angle_error"),
        })
        for episode in episodes:
            hybrid = selected["block_only_torque_readout"][episode]
            baseline = selected["state_action"][episode]
            paired.append({
                "seed": int(payload["protocol"]["seed"]),
                "episode": episode,
                "success_difference": int(bool(hybrid["strict_task_success"])) - int(bool(baseline["strict_task_success"])),
                "block_xy_difference": float(hybrid["block_xy_distance"]) - float(baseline["block_xy_distance"]),
                "angle_difference": float(hybrid["angle_error"]) - float(baseline["angle_error"]),
            })
        for name in arms:
            arms[name].extend(selected[name].values())

    success_diff = [row["success_difference"] for row in paired]
    return {
        "protocol": {
            "artifacts": [str(path) for path in paths],
            "n_seeds": len(paths),
            "n_paired_cases": len(paired),
            "required_fields": sorted(required),
            "protocol_checks": protocol_checks,
        },
        "per_seed": per_seed,
        "pooled": {
            "hybrid_success": int(sum(bool(row["strict_task_success"]) for row in arms["block_only_torque_readout"])),
            "baseline_success": int(sum(bool(row["strict_task_success"]) for row in arms["state_action"])),
            "n": len(paired),
            "hybrid_success_rate": mean([{"x": bool(row["strict_task_success"])} for row in arms["block_only_torque_readout"]], "x"),
            "baseline_success_rate": mean([{"x": bool(row["strict_task_success"]) } for row in arms["state_action"]], "x"),
            "paired_success_gain": sum(success_diff) / max(len(success_diff), 1),
            "hybrid_block_xy": mean(arms["block_only_torque_readout"], "block_xy_distance"),
            "baseline_block_xy": mean(arms["state_action"], "block_xy_distance"),
            "hybrid_angle": mean(arms["block_only_torque_readout"], "angle_error"),
            "baseline_angle": mean(arms["state_action"], "angle_error"),
        },
        "paired_case_records": paired,
        "claim_boundary": (
            "The hybrid arm is a privileged simulator-state task-success result: "
            "the learned adapter proposes actions and a bounded simulator successor "
            "verification commits them. It does not establish that the learned "
            "adapter alone is a stable planner, frozen-LeWM latent causality, "
            "visual online planning, or a universal world model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    report = summarize(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pooled = report["pooled"]
    lines = [
        "# PushT hybrid-planner confirmatory audit",
        "",
        "| arm | strict success | rate | block XY | angle |",
        "|---|---:|---:|---:|---:|",
        f"| hybrid learned-proposal + simulator verification | {pooled['hybrid_success']}/{pooled['n']} | {pooled['hybrid_success_rate']:.4f} | {pooled['hybrid_block_xy']:.3f} | {pooled['hybrid_angle']:.3f} |",
        f"| matched state-action baseline | {pooled['baseline_success']}/{pooled['n']} | {pooled['baseline_success_rate']:.4f} | {pooled['baseline_block_xy']:.3f} | {pooled['baseline_angle']:.3f} |",
        "",
        "| seed | hybrid | baseline | hybrid-baseline |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["per_seed"]:
        lines.append(f"| {row['seed']} | {row['hybrid_success']}/{row['n']} | {row['baseline_success']}/{row['n']} | {row['hybrid_success_rate'] - row['baseline_success_rate']:+.4f} |")
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    args.output.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
