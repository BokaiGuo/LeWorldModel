"""Aggregate matched fresh-seed goal/contact-frame candidate coverage audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/pusht/lewm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    root = Path(args.root)
    folders = [
        "goal_aligned_candidate_coverage_seed20261101",
        "goal_aligned_candidate_coverage_seed20261102",
    ]
    reports = [json.loads((root / folder / "goal_aligned_candidate_coverage.json").read_text()) for folder in folders]
    for report in reports:
        if not report["sanity_checks"]["all_finite"]:
            raise ValueError("non-finite source report")

    metrics = (
        "candidate_count", "oracle_available", "oracle_count", "selected_feasible",
        "top8_hit", "top32_hit", "selected_true_score", "oracle_best_score",
        "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress",
        "selected_contact",
    )
    summaries = {}
    for arm_index, arm in enumerate(("absolute_fan", "goal_contact_frame", "hybrid")):
        summaries[arm] = {
            metric: bootstrap(
                np.concatenate([np.asarray([row[metric] for row in report["rows"][arm]], dtype=np.float64) for report in reports]),
                args.bootstrap, 100 + arm_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    paired = {}
    for arm_index, arm in enumerate(("goal_contact_frame", "hybrid")):
        paired[arm] = {
            metric: bootstrap(
                np.concatenate([
                    np.asarray([new[metric] - old[metric] for old, new in zip(report["rows"]["absolute_fan"], report["rows"][arm])], dtype=np.float64)
                    for report in reports
                ]),
                args.bootstrap, 500 + arm_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    report = {
        "protocol": {
            "source_seeds": [20261101, 20261102],
            "source_folders": folders,
            "held_out_groups_per_seed": 64,
            "pooled_matched_groups": 128,
            "all_runtime_successor_queries": False,
            "simulator_successor_used_for_labels_only": True,
        },
        "pooled_summaries": summaries,
        "paired_vs_absolute": paired,
        "claim_boundary": (
            "Across two fresh seeds, a current-state/goal-conditioned candidate bank increases "
            "oracle availability relative to the historical absolute fan, but the frozen ranker "
            "does not convert that increase into selected feasibility or contact preservation. "
            "This is a privileged simulator-state interface diagnostic, not visual planning or "
            "task-success evidence."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "goal_aligned_candidate_coverage_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    goal = summaries["goal_contact_frame"]
    delta = paired["goal_contact_frame"]
    lines = [
        "# Goal/contact-frame candidate coverage summary", "",
        "| arm | candidates | oracle availability | selected feasible | top-8 hit | selected true rank |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("absolute_fan", "goal_contact_frame", "hybrid"):
        values = summaries[arm]
        lines.append(
            "| %s | %.1f | %.3f | %.3f | %.3f | %.1f |" % (
                arm, values["candidate_count"]["estimate"], values["oracle_available"]["estimate"],
                values["selected_feasible"]["estimate"], values["top8_hit"]["estimate"],
                values["selected_rank_by_true_score"]["estimate"],
            )
        )
    lines += [
        "", "Paired goal/contact-frame minus absolute-fan changes:",
        "",
        "- oracle availability: %.3f [%.3f, %.3f]" % (delta["oracle_available"]["estimate"], delta["oracle_available"]["ci95_low"], delta["oracle_available"]["ci95_high"]),
        "- selected feasibility: %.3f [%.3f, %.3f]" % (delta["selected_feasible"]["estimate"], delta["selected_feasible"]["ci95_low"], delta["selected_feasible"]["ci95_high"]),
        "- selected contact: %.3f [%.3f, %.3f]" % (delta["selected_contact"]["estimate"], delta["selected_contact"]["ci95_low"], delta["selected_contact"]["ci95_high"]),
        "",
        report["claim_boundary"], "",
    ]
    (output / "goal_aligned_candidate_coverage_summary.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
