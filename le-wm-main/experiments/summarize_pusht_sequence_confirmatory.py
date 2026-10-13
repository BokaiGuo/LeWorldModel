"""Aggregate matched pure-learned two-action PushT confirmation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "object_centric_learned_planner.json").read_text())
    if not payload.get("claim_boundary", "").startswith("Privileged simulator-state"):
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
        "planning_objective": "staged_contact_direct_sequence_readout",
        "training_protocol": "contact_intervention",
        "direct_state_pool": "protocol",
        "direct_train_states": 128,
        "direct_sequence_action_count": 25,
        "direct_sequence_count": 625,
        "action_block": 2,
        "episodes": 8,
        "max_steps": 60,
        "same_cases": True,
        "same_candidate_bank_seed_schedule": True,
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
    paired: list[dict[str, Any]] = []
    per_seed = []
    for payload in payloads:
        rows = payload["rows"]
        selected = {
            name: {int(row["episode"]): row for row in rows if row["model"] == name}
            for name in arms
        }
        if any(len(values) != 8 for values in selected.values()):
            raise ValueError("expected 8 rows per arm and seed")
        episodes = sorted(selected["state_action"])
        if episodes != sorted(selected["block_only_torque_readout"]):
            raise ValueError("unmatched episodes within seed")
        learned = [selected["block_only_torque_readout"][episode] for episode in episodes]
        baseline = [selected["state_action"][episode] for episode in episodes]
        per_seed.append({
            "seed": int(payload["protocol"]["seed"]),
            "n": len(episodes),
            "learned_success": int(sum(bool(row["strict_task_success"]) for row in learned)),
            "baseline_success": int(sum(bool(row["strict_task_success"]) for row in baseline)),
            "learned_block_xy": mean(learned, "block_xy_distance"),
            "baseline_block_xy": mean(baseline, "block_xy_distance"),
            "learned_angle": mean(learned, "angle_error"),
            "baseline_angle": mean(baseline, "angle_error"),
        })
        for episode in episodes:
            candidate = selected["block_only_torque_readout"][episode]
            reference = selected["state_action"][episode]
            paired.append({
                "seed": int(payload["protocol"]["seed"]),
                "episode": episode,
                "success_difference": int(bool(candidate["strict_task_success"])) - int(bool(reference["strict_task_success"])),
                "block_xy_difference": float(candidate["block_xy_distance"]) - float(reference["block_xy_distance"]),
                "angle_difference": float(candidate["angle_error"]) - float(reference["angle_error"]),
            })
        for name in arms:
            arms[name].extend(selected[name].values())

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
            "learned_success": int(sum(bool(row["strict_task_success"]) for row in arms["block_only_torque_readout"])),
            "baseline_success": int(sum(bool(row["strict_task_success"]) for row in arms["state_action"])),
            "n": len(paired),
            "learned_success_rate": mean([{"x": bool(row["strict_task_success"])} for row in arms["block_only_torque_readout"]], "x"),
            "baseline_success_rate": mean([{"x": bool(row["strict_task_success"])} for row in arms["state_action"]], "x"),
            "paired_success_gain": sum(row["success_difference"] for row in paired) / max(len(paired), 1),
            "learned_block_xy": mean(arms["block_only_torque_readout"], "block_xy_distance"),
            "baseline_block_xy": mean(arms["state_action"], "block_xy_distance"),
            "learned_angle": mean(arms["block_only_torque_readout"], "angle_error"),
            "baseline_angle": mean(arms["state_action"], "angle_error"),
        },
        "paired_case_records": paired,
        "claim_boundary": (
            "This is a privileged simulator-state pure-learned two-action diagnostic. "
            "The endpoint map is trained offline with simulator labels and selected "
            "without runtime successor verification. The pooled result does not establish "
            "stable task success, frozen-LeWM latent causality, visual online planning, "
            "or a universal world model."
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
        "# PushT pure-learned sequence confirmation audit",
        "",
        "| arm | strict success | rate | block XY | angle |",
        "|---|---:|---:|---:|---:|",
        f"| learned two-action endpoint | {pooled['learned_success']}/{pooled['n']} | {pooled['learned_success_rate']:.4f} | {pooled['learned_block_xy']:.3f} | {pooled['learned_angle']:.3f} |",
        f"| matched state-action baseline | {pooled['baseline_success']}/{pooled['n']} | {pooled['baseline_success_rate']:.4f} | {pooled['baseline_block_xy']:.3f} | {pooled['baseline_angle']:.3f} |",
        "",
        "| seed | learned | baseline | learned-baseline |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['learned_success']}/{row['n']} | "
            f"{row['baseline_success']}/{row['n']} | "
            f"{row['learned_success']/row['n'] - row['baseline_success']/row['n']:+.4f} |"
        )
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    args.output.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
