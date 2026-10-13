"""Paired multi-seed summary for the contact history/velocity diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAMES = ("state_action", "object_centric", "history")
METRICS = ("block_xy_error", "angle_error", "separation_abs_error")


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n": int(len(values)),
    }


def paired_bootstrap_by_seed(
    differences: list[np.ndarray], draws: int, seed: int
) -> dict[str, Any]:
    """Bootstrap cases within each seed, preserving the seed clusters."""
    arrays = [np.asarray(item, dtype=np.float64) for item in differences]
    if any(len(item) == 0 for item in arrays):
        raise ValueError("cannot bootstrap an empty seed cell")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        resampled = [item[rng.integers(0, len(item), size=len(item))] for item in arrays]
        samples.append(float(np.concatenate(resampled).mean()))
    values = np.concatenate(arrays)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n": int(len(values)),
        "n_seeds": int(len(arrays)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261010)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads((path / "contact_history_velocity_diagnostic.json").read_text())


def main() -> None:
    args = parse_args()
    reports = [load_report(path) for path in args.inputs]
    indexed = []
    for report in reports:
        indexed.append({
            (item["model"], int(item["horizon"])): item
            for item in report["reports"]
        })
    horizons = sorted({int(item["horizon"]) for item in reports[0]["reports"]})
    comparisons = []
    for horizon in horizons:
        for target in ("object_centric", "history"):
            for metric in METRICS:
                per_seed = []
                for seed_index, report_index in enumerate(indexed):
                    baseline = report_index[("state_action", horizon)]["rows"]
                    candidate = report_index[(target, horizon)]["rows"]
                    if len(baseline) != len(candidate):
                        raise ValueError("unmatched case counts")
                    baseline_values = np.asarray([
                        float(np.mean(row[metric])) if metric in ("block_xy_error", "angle_error")
                        else float(row[metric])
                        for row in baseline
                    ])
                    candidate_values = np.asarray([
                        float(np.mean(row[metric])) if metric in ("block_xy_error", "angle_error")
                        else float(row[metric])
                        for row in candidate
                    ])
                    per_seed.append(candidate_values - baseline_values)
                comparisons.append({
                    "target": target,
                    "baseline": "state_action",
                    "horizon": horizon,
                    "metric": metric,
                    "candidate_minus_baseline": paired_bootstrap_by_seed(
                        per_seed, args.draws, args.seed + horizon + len(target) + len(metric)
                    ),
                    "per_seed": [
                        bootstrap(values, args.draws, args.seed + index + 1000)
                        for index, values in enumerate(per_seed)
                    ],
                })

    mode_summary = []
    for report_index, report in zip(indexed, reports):
        for horizon in horizons:
            rows = report_index[("state_action", horizon)]["rows"]
            for mode in ("free", "approach", "contact"):
                subset = [row for row in rows if row["mode"] == mode]
                if not subset:
                    continue
                mode_summary.append({
                    "seed": report["protocol"]["seed"],
                    "horizon": horizon,
                    "mode": mode,
                    "n": len(subset),
                    "mean_state_action_block_xy_error": float(np.mean([
                        np.mean(row["block_xy_error"]) for row in subset
                    ])),
                    "mean_contact_distance": float(np.mean([
                        row["contact_distance"] for row in subset
                    ])),
                    "mean_action_tangent": float(np.mean([
                        row["action_tangent"] for row in subset
                    ])),
                    "mean_agent_speed": float(np.mean([
                        row["agent_speed"] for row in subset
                    ])),
                })

    result = {
        "protocol": {
            "inputs": [str(path) for path in args.inputs],
            "paired_unit": "same held-out state and +a/-a/zero intervention within each seed",
            "bootstrap": "case resampling within seed clusters",
            "draws": args.draws,
            "seed": args.seed,
        },
        "comparisons": comparisons,
        "mode_summary": mode_summary,
        "claim_boundary": (
            "Privileged simulator-state physical-adapter diagnosis. A paired gain identifies predictive input "
            "sufficiency for this simulator protocol; it does not establish frozen-LeWM latent causality, visual "
            "online planning, or general PushT task performance."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "contact_history_velocity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    lines = [
        "# PushT contact history and velocity paired summary",
        "",
        "Differences are candidate minus state_action; negative values are improvements.",
        "Intervals are paired bootstrap 95% CIs with case resampling within each seed.",
        "",
        "| candidate | horizon | metric | difference |",
        "|---|---:|---|---:|",
    ]
    for item in comparisons:
        value = item["candidate_minus_baseline"]
        lines.append(
            "| %s | %d | %s | %.3f [%.3f, %.3f] |" % (
                item["target"], item["horizon"], item["metric"],
                value["estimate"], value["ci95_low"], value["ci95_high"],
            )
        )
    lines += ["", "## Mode-conditioned baseline diagnostics", ""]
    lines += [
        "| seed | horizon | mode | n | baseline XY error | contact distance | action tangent | agent speed |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in mode_summary:
        lines.append(
            "| %s | %d | %s | %d | %.3f | %.3f | %.3f | %.3f |" % (
                item["seed"], item["horizon"], item["mode"], item["n"],
                item["mean_state_action_block_xy_error"], item["mean_contact_distance"],
                item["mean_action_tangent"], item["mean_agent_speed"],
            )
        )
    lines += ["", result["claim_boundary"], ""]
    (args.output / "contact_history_velocity_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
