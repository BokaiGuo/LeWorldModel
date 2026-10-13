"""Audit a paired RGB-only orientation-expert repair without hiding failures.

The script compares immutable h=2 torque experts against one h=1 controller
on the *same declared rendered-RGB episodes*.  It reports both final task
metrics and how frequently the orientation phase was actually invoked.  The
episode is the paired unit for the reported bootstrap intervals; different
expert checkpoints are kept separate and are never pooled as independent
environment seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("protocol", {})
    if not protocol.get("checkpoint_immutable", False):
        raise ValueError(f"{path}: checkpoint must be immutable")
    if protocol.get("runtime_state_access", True) or protocol.get("runtime_successor_verification", True):
        raise ValueError(f"{path}: runtime must be RGB-only without successor verification")
    rows = payload.get("planning", {}).get("episodes", [])
    if len(rows) < 2:
        raise ValueError(f"{path}: requires at least two episodes")
    return payload


def paired_bootstrap(delta: np.ndarray, seed: int, draws: int = 20_000) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "n_paired_episodes": int(len(delta)),
        "bootstrap_unit": "episode (paired; checkpoint fixed)",
    }


def phase_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expert_steps = np.asarray([
        sum(item.get("action_source") == "orientation_expert" for item in row["trace"])
        for row in rows
    ], dtype=np.int64)
    switched = np.asarray([
        sum(
            left.get("action_source") != right.get("action_source")
            for left, right in zip(row["trace"], row["trace"][1:])
        )
        for row in rows
    ], dtype=np.int64)
    used = expert_steps > 0
    escaped = np.asarray([float(row["block_xy_distance"]) > 35.0 for row in rows], dtype=bool)
    return {
        "episodes_using_orientation_expert": int(used.sum()),
        "mean_orientation_expert_steps": float(expert_steps.mean()),
        "mean_controller_switches": float(switched.mean()),
        "expert_used_then_final_xy_outside_35px": int(np.logical_and(used, escaped).sum()),
    }


def compare(baseline: dict[str, Any], expert: dict[str, Any], label: str, seed: int) -> dict[str, Any]:
    base_protocol, expert_protocol = baseline["protocol"], expert["protocol"]
    if base_protocol.get("eval_seed") != expert_protocol.get("eval_seed"):
        raise ValueError(f"{label}: eval seed mismatch")
    base_rows, expert_rows = baseline["planning"]["episodes"], expert["planning"]["episodes"]
    if [row["episode"] for row in base_rows] != [row["episode"] for row in expert_rows]:
        raise ValueError(f"{label}: episode identities mismatch")
    success_delta = np.asarray([int(row["strict_success"]) for row in expert_rows]) - np.asarray(
        [int(row["strict_success"]) for row in base_rows]
    )
    xy_delta = np.asarray([row["block_xy_distance"] for row in expert_rows]) - np.asarray(
        [row["block_xy_distance"] for row in base_rows]
    )
    angle_delta = np.asarray([row["angle_error"] for row in expert_rows]) - np.asarray(
        [row["angle_error"] for row in base_rows]
    )
    return {
        "expert": label,
        "paired_deltas_expert_minus_base": {
            "strict_success": paired_bootstrap(success_delta, seed),
            "block_xy_distance": paired_bootstrap(xy_delta, seed + 1),
            "angle_error": paired_bootstrap(angle_delta, seed + 2),
            "expert_only_successes": int((success_delta > 0).sum()),
            "base_only_successes": int((success_delta < 0).sum()),
        },
        "orientation_phase_usage": phase_usage(expert_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expert", type=str, nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = load(args.baseline)
    comparisons = []
    for index, spec in enumerate(args.expert):
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--expert entries must be LABEL=PATH")
        comparisons.append(compare(baseline, load(Path(raw_path)), label, 20411 + 10 * index))
    base_rows = baseline["planning"]["episodes"]
    report = {
        "artifact": "paired_rgb_orientation_phase_repair_audit",
        "protocol": {
            "eval_seed": baseline["protocol"].get("eval_seed"),
            "episodes": len(base_rows),
            "runtime_boundary": "rendered RGB history and goal only; state evaluator-side",
            "baseline": str(args.baseline.resolve()),
            "reversible_orientation_phase": bool(baseline["protocol"]["interface"].get("reversible_orientation_phase", False)),
        },
        "baseline": {
            "strict_success_count": int(sum(row["strict_success"] for row in base_rows)),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in base_rows])),
            "mean_angle_error": float(np.mean([row["angle_error"] for row in base_rows])),
        },
        "comparisons": comparisons,
        "claim_boundary": (
            "A paired interface audit, not evidence of stable visual planning or a universal world model. "
            "A favorable episode-level delta cannot replace independent checkpoint-seed replication."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "orientation_phase_repair_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Paired RGB orientation-phase repair audit", "",
        f"- Shared held-out evaluation seed: `{report['protocol']['eval_seed']}`.",
        f"- Baseline: {report['baseline']['strict_success_count']}/{report['protocol']['episodes']}; "
        f"XY {report['baseline']['mean_block_xy_distance']:.3f}; angle {report['baseline']['mean_angle_error']:.3f}.", "",
        "| expert | success delta | XY delta | angle delta | expert-used episodes | escaped after expert |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        delta, usage = row["paired_deltas_expert_minus_base"], row["orientation_phase_usage"]
        lines.append(
            f"| {row['expert']} | {delta['strict_success']['mean']:.4f} | {delta['block_xy_distance']['mean']:.3f} | "
            f"{delta['angle_error']['mean']:.3f} | {usage['episodes_using_orientation_expert']} | "
            f"{usage['expert_used_then_final_xy_outside_35px']} |"
        )
    (args.output / "orientation_phase_repair_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
