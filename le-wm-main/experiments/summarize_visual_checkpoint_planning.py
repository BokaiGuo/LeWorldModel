"""Aggregate multi-seed RGB-only checkpoint planning evaluations.

The evaluator uses simulator state only for scoring.  This summarizer keeps
the seed as the uncertainty unit and refuses to call a result stable when a
seed is missing, when the protocol is not matched, or when successor
verification/privileged inputs were enabled.  It is intentionally a reporting
surface, not a success-promoting filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not payload.get("protocol", {}).get("checkpoint_immutable", False):
        raise ValueError(f"{path}: checkpoint is not declared immutable")
    protocol = payload["protocol"]
    if protocol.get("runtime_state_access", True):
        raise ValueError(f"{path}: runtime state access is enabled")
    if protocol.get("runtime_successor_verification", True):
        raise ValueError(f"{path}: runtime successor verification is enabled")
    episodes = payload.get("planning", {}).get("episodes", [])
    if not episodes:
        raise ValueError(f"{path}: no planning episodes")
    return payload


def _bootstrap(values: np.ndarray, seed: int, draws: int = 20000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two seed clusters are required")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "n_seed_clusters": int(values.size),
        "bootstrap_unit": "seed",
        "draws": int(draws),
    }


def summarize(paths: list[Path], *, expected_seeds: list[int] | None = None) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("multi-seed aggregation requires at least two artifacts")
    reports = [_load(path) for path in paths]
    seed_rows: list[dict[str, Any]] = []
    protocol_keys = ("interface",)
    reference = reports[0]["protocol"]
    for path, report in zip(paths, reports):
        protocol = report["protocol"]
        for key in protocol_keys:
            if protocol.get(key) != reference.get(key):
                raise ValueError(f"protocol mismatch for {path}: {key}")
        episodes = report["planning"]["episodes"]
        seed = int(path.parent.name.split("_")[-1]) if path.parent.name.startswith("seed_") else int(protocol.get("seed", -1))
        success = np.asarray([float(row["strict_success"]) for row in episodes], dtype=np.float64)
        xy = np.asarray([float(row["block_xy_distance"]) for row in episodes], dtype=np.float64)
        angle = np.asarray([float(row["angle_error"]) for row in episodes], dtype=np.float64)
        if not (np.isfinite(success).all() and np.isfinite(xy).all() and np.isfinite(angle).all()):
            raise ValueError(f"{path}: non-finite episode metrics")
        seed_rows.append({
            "seed": seed,
            "episodes": int(len(episodes)),
            "strict_success": float(success.mean()),
            "strict_success_count": int(success.sum()),
            "mean_block_xy_distance": float(xy.mean()),
            "mean_angle_error": float(angle.mean()),
        })
    observed = sorted(row["seed"] for row in seed_rows)
    if expected_seeds is not None and observed != sorted(int(x) for x in expected_seeds):
        raise ValueError(f"seed coverage mismatch: observed={observed}, expected={sorted(expected_seeds)}")
    success = np.asarray([row["strict_success"] for row in seed_rows])
    xy = np.asarray([row["mean_block_xy_distance"] for row in seed_rows])
    angle = np.asarray([row["mean_angle_error"] for row in seed_rows])
    report = {
        "artifact": "multi_seed_visual_checkpoint_planning_summary",
        "protocol": {
            "seed_clusters": observed,
            "n_seed_clusters": len(seed_rows),
            "episodes_per_seed": [row["episodes"] for row in seed_rows],
            "uncertainty_unit": "seed; episodes averaged within seed",
            "runtime_boundary": "RGB/history/goal/action only; simulator state evaluator-side",
            "checkpoint_immutable": True,
            "runtime_successor_verification": False,
        },
        "seed_rows": seed_rows,
        "aggregate": {
            "strict_success_rate": _bootstrap(success, 20261009),
            "mean_block_xy_distance": _bootstrap(xy, 20261010),
            "mean_angle_error": _bootstrap(angle, 20261011),
        },
        "acceptance": {
            "prespecified_min_strict_success_rate_per_seed": 0.50,
            "prespecified_stability_rule": "every seed must reach at least 50% strict success on 100 fresh episodes; the seed-bootstrap CI is reported and no snapshot is selected by test performance",
            "all_seeds_meet_threshold": bool(np.all(success >= 0.50)),
            "status": "stable_candidate" if bool(np.all(success >= 0.50)) else "negative_or_unstable_boundary",
        },
        "claim_boundary": [
            "This is a multi-seed RGB-only planning evaluation of the saved causal visual checkpoints, not frozen LeWM evidence.",
            "The summary does not establish a universal world model, physical causality, or cross-environment transfer.",
            "A positive status requires the full paired baseline matrix and independent test environments; this artifact alone cannot promote that claim.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="*", type=int)
    args = parser.parse_args()
    report = summarize(args.inputs, expected_seeds=args.expected_seeds or None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Multi-seed RGB-only checkpoint planning summary", "",
        "| seed | episodes | strict success | block XY | angle error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["seed_rows"]:
        lines.append(f"| {row['seed']} | {row['episodes']} | {row['strict_success_count']}/{row['episodes']} ({row['strict_success']:.3f}) | {row['mean_block_xy_distance']:.3f} | {row['mean_angle_error']:.3f} |")
    lines.extend(["", f"Status: `{report['acceptance']['status']}`.", "", *[f"- {x}" for x in report["claim_boundary"]]])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
