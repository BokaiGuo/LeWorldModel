"""Summarize the restored-state terminal-stabilization audit.

This report deliberately separates strict task completion from the safety
estimand.  The latter checks that the lock state never invokes acquisition and
that terminal return preserves the already-closed block pose.  It does not
reinterpret a safe hold as a successful learned torque controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline8421", required=True)
    parser.add_argument("--baseline20261021", required=True)
    parser.add_argument("--terminal8421", required=True)
    parser.add_argument("--terminal20261021", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(Path(path).read_text())
    rows = [row for row in report["rows"] if row["model"] == "block_only_torque_readout"]
    if not rows:
        raise ValueError(f"no torque-readout rows in {path}")
    return report, rows


def arm_metrics(rows: list[dict[str, Any]], terminal: bool) -> dict[str, Any]:
    lock_steps = 0
    acquire_in_lock = 0
    learned_tube_violations = 0
    terminal_returns = 0
    terminal_return_preserved = 0
    ever_closed_failures = 0
    fallback_counts: dict[str, int] = {}
    for row in rows:
        trace = row["trace"]
        minimum_xy = min(
            [float(item["block_xy_after"]) for item in trace if "block_xy_after" in item]
            + [float(row["block_xy_distance"])]
        )
        if minimum_xy < 20.0 and not row["success"]:
            ever_closed_failures += 1
        for item in trace:
            mode = item.get("selection_mode")
            if mode is not None:
                fallback_counts[str(mode)] = fallback_counts.get(str(mode), 0) + 1
            if item.get("closure_translation_lock_before"):
                lock_steps += 1
                acquire_in_lock += int(item.get("phase") == "acquire")
                if (
                    item.get("phase") == "verified_torque_effect"
                    and float(item["block_xy_after"]) > 22.0 + 1e-9
                ):
                    learned_tube_violations += 1
            if item.get("phase") == "terminal_return_preserving":
                terminal_returns += 1
                terminal_return_preserved += int(
                    float(item["block_xy_after"]) < 20.0
                    and float(item["angle_after"]) < np.pi / 9.0
                )
    return {
        "n": len(rows),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "mean_block_xy": float(np.mean([row["block_xy_distance"] for row in rows])),
        "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
        "mean_agent_xy": float(np.mean([row["agent_xy_distance"] for row in rows])),
        "lock_steps": lock_steps,
        "acquire_in_lock": acquire_in_lock,
        "learned_tube_violations": learned_tube_violations,
        "terminal_returns": terminal_returns,
        "terminal_return_preserved": terminal_return_preserved,
        "ever_closed_then_failed": ever_closed_failures,
        "fallback_selection_counts": fallback_counts,
        "terminal_arm": terminal,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for seed_name, baseline_path, terminal_path in (
        ("seed8421", args.baseline8421, args.terminal8421),
        ("seed20261021", args.baseline20261021, args.terminal20261021),
    ):
        baseline_report, baseline_rows = load_rows(baseline_path)
        terminal_report, terminal_rows = load_rows(terminal_path)
        if baseline_report["protocol"]["seed"] != terminal_report["protocol"]["seed"]:
            raise ValueError(f"seed mismatch for {seed_name}")
        cases[seed_name] = {
            "baseline": arm_metrics(baseline_rows, terminal=False),
            "terminal_stabilized": arm_metrics(terminal_rows, terminal=True),
        }

    baseline_rows = [
        cases[name]["baseline"] for name in sorted(cases)
    ]
    terminal_rows = [
        cases[name]["terminal_stabilized"] for name in sorted(cases)
    ]
    summary = {
        "n_cases": int(sum(row["n"] for row in baseline_rows)),
        "baseline_successes": int(sum(row["successes"] for row in baseline_rows)),
        "terminal_successes": int(sum(row["successes"] for row in terminal_rows)),
        "baseline_mean_block_xy": float(np.average(
            [row["mean_block_xy"] for row in baseline_rows],
            weights=[row["n"] for row in baseline_rows],
        )),
        "terminal_mean_block_xy": float(np.average(
            [row["mean_block_xy"] for row in terminal_rows],
            weights=[row["n"] for row in terminal_rows],
        )),
        "baseline_mean_angle_error": float(np.average(
            [row["mean_angle_error"] for row in baseline_rows],
            weights=[row["n"] for row in baseline_rows],
        )),
        "terminal_mean_angle_error": float(np.average(
            [row["mean_angle_error"] for row in terminal_rows],
            weights=[row["n"] for row in terminal_rows],
        )),
        "terminal_lock_acquire_events": int(sum(row["acquire_in_lock"] for row in terminal_rows)),
        "terminal_return_preservation": [
            int(row["terminal_return_preserved"]) for row in terminal_rows
        ],
        "terminal_ever_closed_then_failed": int(sum(row["ever_closed_then_failed"] for row in terminal_rows)),
        "claim_boundary": (
            "Privileged simulator-state learned-planner audit. The terminal arm tests state-consistent one-step proposal verification, "
            "translation hysteresis, inertia-aware hold fallback, and a separate closure-preserving return phase. It does not establish "
            "frozen-LeWM visual planning, learned physical counterfactual accuracy, or general PushT performance."
        ),
    }
    result = {"by_seed": cases, "summary": summary}
    (output / "terminal_stabilization_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    lines = [
        "# PushT terminal stabilization audit",
        "",
        "| seed | arm | success | mean block XY | mean angle | mean agent XY | lock acquire | terminal return preserved |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_name in sorted(cases):
        for arm_name in ("baseline", "terminal_stabilized"):
            row = cases[seed_name][arm_name]
            lines.append(
                "| %s | %s | %d/%d | %.3f | %.3f | %.3f | %d | %d/%d |"
                % (
                    seed_name,
                    arm_name,
                    row["successes"],
                    row["n"],
                    row["mean_block_xy"],
                    row["mean_angle_error"],
                    row["mean_agent_xy"],
                    row["acquire_in_lock"],
                    row["terminal_return_preserved"],
                    row["terminal_returns"],
                )
            )
    lines += ["", summary["claim_boundary"], ""]
    (output / "terminal_stabilization_summary.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
