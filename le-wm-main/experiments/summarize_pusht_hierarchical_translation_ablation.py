"""Summarize the hierarchical translation/torque ablation.

The hierarchical arm is intentionally a diagnostic: it uses a learned
block-pose adapter for long-range translation and the verified local torque
readout only inside the near-goal tube.  This script keeps it separate from
the corrected top-eight verification arm and reports the result as a paired
negative ablation when it does not improve strict closure.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline8421", type=Path, required=True)
    parser.add_argument("--baseline20261021", type=Path, required=True)
    parser.add_argument("--hierarchical8421", type=Path, required=True)
    parser.add_argument("--hierarchical20261021", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--candidate-label", default="hierarchical")
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text())
    rows = [row for row in report["rows"] if row["model"] == "block_only_torque_readout"]
    if not rows:
        raise ValueError(f"no block_only_torque_readout rows in {path}")
    return rows


def bootstrap(delta: np.ndarray, draws: int, seed: int) -> dict[str, float | int]:
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(delta)),
    }


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(item.get("phase") for row in rows for item in row["trace"])
    return {
        "n": len(rows),
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "mean_block_xy": float(np.mean([row["block_xy_distance"] for row in rows])),
        "mean_angle_error": float(np.mean([row["angle_error"] for row in rows])),
        "mean_agent_xy": float(np.mean([row["agent_xy_distance"] for row in rows])),
        "phase_counts": dict(sorted((str(k), int(v)) for k, v in phases.items())),
        "all_finite": bool(all(
            np.isfinite(row[key])
            for row in rows
            for key in ("block_xy_distance", "angle_error", "agent_xy_distance", "return")
        )),
    }


def main() -> None:
    args = parse_args()
    pairs = []
    by_seed = {}
    for name, baseline_path, hierarchical_path in (
        ("seed8421", args.baseline8421, args.hierarchical8421),
        ("seed20261021", args.baseline20261021, args.hierarchical20261021),
    ):
        baseline = load(baseline_path)
        hierarchical = load(hierarchical_path)
        if len(baseline) != len(hierarchical):
            raise ValueError(f"unmatched row count for {name}")
        by_seed[name] = {
            "baseline": arm_summary(baseline),
            args.candidate_label: arm_summary(hierarchical),
        }
        pairs.extend(zip(baseline, hierarchical))

    metrics = {
        "success": lambda row: float(bool(row["success"])),
        "block_xy_distance": lambda row: float(row["block_xy_distance"]),
        "angle_error": lambda row: float(row["angle_error"]),
        "agent_xy_distance": lambda row: float(row["agent_xy_distance"]),
    }
    comparisons = {}
    for name, getter in metrics.items():
        baseline = np.asarray([getter(pair[0]) for pair in pairs], dtype=np.float64)
        candidate = np.asarray([getter(pair[1]) for pair in pairs], dtype=np.float64)
        comparisons[name] = {
            "baseline_mean": float(baseline.mean()),
            f"{args.candidate_label}_mean": float(candidate.mean()),
            f"{args.candidate_label}_minus_baseline": bootstrap(
                candidate - baseline, args.bootstrap, args.seed + len(comparisons)
            ),
        }

    result = {
        "protocol": {
            "pairing": "same case index within each independent seed run",
            "n": len(pairs),
            "baseline": "corrected state-restored top-eight verification",
            args.candidate_label: (
                "learned block-pose translation for block-XY >= 45 px, "
                "followed by state-consistent verification of the local action fan"
            ),
        },
        "by_seed": by_seed,
        "comparisons": comparisons,
        "claim_boundary": (
            "Privileged simulator-state negative ablation. The candidate "
            "stage separation is finite and state-consistent, but does not "
            "improve strict learned-planner closure over corrected verification. "
            "The validated objective repair remains the simulator-in-the-loop "
            "receding block-pose planner with a separate closure-preserving return."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stem = (
        "hierarchical_translation_ablation"
        if args.candidate_label == "hierarchical"
        else f"{args.candidate_label}_ablation"
    )
    (args.output / f"{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    lines = [
        f"# {args.candidate_label} verification ablation",
        "",
        "| seed | arm | success | block XY | angle | agent XY | phases |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for seed in ("seed20261021", "seed8421"):
        for arm in ("baseline", args.candidate_label):
            row = by_seed[seed][arm]
            lines.append(
                f"| {seed} | {arm} | {row['successes']}/{row['n']} | "
                f"{row['mean_block_xy']:.3f} | {row['mean_angle_error']:.3f} | "
                f"{row['mean_agent_xy']:.3f} | {row['phase_counts']} |"
            )
    lines += ["", result["claim_boundary"], ""]
    (args.output / f"{stem}.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
