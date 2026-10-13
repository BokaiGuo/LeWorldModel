"""Aggregate candidate-generation, ranker-shift, and closure evidence."""

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


def read(root: Path, folder: str, filename: str) -> dict[str, Any]:
    report = json.loads((root / folder / filename).read_text())
    if not report["sanity_checks"]["all_finite"]:
        raise ValueError(f"non-finite artifact: {folder}")
    return report


def paired_reports(
    reports: list[dict[str, Any]],
    new_arm: str,
    old_arm: str,
    metrics: tuple[str, ...],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for metric_index, metric in enumerate(metrics):
        values = np.concatenate([
            np.asarray([new[metric] - old[metric] for old, new in zip(report["rows"][old_arm], report["rows"][new_arm])], dtype=np.float64)
            for report in reports
        ])
        output[metric] = bootstrap(values, draws, seed + metric_index)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/pusht/lewm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    root = Path(args.root)

    generation = [
        read(root, "candidate_generation_intervention_seed20261093", "candidate_generation_intervention.json"),
        read(root, "candidate_generation_intervention_seed20261094", "candidate_generation_intervention.json"),
    ]
    repeated = read(root, "candidate_generation_intervention_repeat2_seed20261095", "candidate_generation_intervention.json")
    shift = [
        read(root, "candidate_training_shift_seed20261096", "candidate_training_shift.json"),
        read(root, "candidate_training_shift_seed20261097", "candidate_training_shift.json"),
    ]
    closure = [
        read(root, "candidate_augmented_closure_balanced_seed20261098", "candidate_augmented_closure.json"),
        read(root, "candidate_augmented_closure_balanced_seed20261099", "candidate_augmented_closure.json"),
    ]

    generation_summary = {}
    for variant in ("baseline_fan", "dense_fan", "goal_templates", "dense_plus_templates"):
        generation_summary[variant] = {
            metric: bootstrap(
                np.concatenate([
                    np.asarray([row[metric] for row in report["rows"][variant]], dtype=np.float64)
                    for report in generation
                ]),
                args.bootstrap,
                100 + index,
            )
            for index, metric in enumerate(("candidate_count", "oracle_available", "selected_feasible", "top8_hit", "selected_angle_progress", "selected_xy_progress"))
        }

    report = {
        "protocol": {
            "source_generation_seeds": 2,
            "source_training_shift_seeds": 2,
            "source_balanced_closure_seeds": 2,
            "all_runtime_candidate_successor_verification": False,
        },
        "two_step_candidate_generation": {
            "pooled_summaries": generation_summary,
            "dense_minus_baseline": paired_reports(
                generation, "dense_fan", "baseline_fan",
                ("oracle_available", "selected_feasible", "top8_hit", "selected_angle_progress", "selected_xy_progress"),
                args.bootstrap, 200,
            ),
            "templates_minus_baseline": paired_reports(
                generation, "goal_templates", "baseline_fan",
                ("oracle_available", "selected_feasible", "top8_hit", "selected_angle_progress", "selected_xy_progress"),
                args.bootstrap, 300,
            ),
        },
        "repeat2_candidate_generation": {
            "summaries": repeated["summaries"],
            "dense_minus_baseline": paired_reports(
                [repeated], "dense_fan", "baseline_fan",
                ("oracle_available", "selected_feasible", "top8_hit", "selected_angle_progress", "selected_xy_progress"),
                args.bootstrap, 400,
            ),
        },
        "ranker_distribution_matching": {
            "dense_ranker_minus_baseline_ranker_on_dense": paired_reports(
                shift, "dense_ranker_on_dense", "baseline_ranker_on_dense",
                ("selected_feasible", "top8_hit", "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress"),
                args.bootstrap, 500,
            ),
            "dense_ranker_dense_minus_baseline_ranker_baseline": paired_reports(
                shift, "dense_ranker_on_dense", "baseline_ranker_on_baseline",
                ("selected_feasible", "top8_hit", "selected_rank_by_true_score", "selected_angle_progress", "selected_xy_progress"),
                args.bootstrap, 600,
            ),
        },
        "closed_loop_balanced_utility": {
            "dense_ranker_dense_minus_baseline_ranker_baseline": paired_reports(
                closure, "dense_ranker_dense_fan", "baseline_ranker_baseline_fan",
                ("strict_success", "final_xy", "final_angle", "cycles"),
                args.bootstrap, 700,
            ),
            "baseline_ranker_dense_minus_baseline_ranker_baseline": paired_reports(
                closure, "baseline_ranker_dense_fan", "baseline_ranker_baseline_fan",
                ("strict_success", "final_xy", "final_angle", "cycles"),
                args.bootstrap, 800,
            ),
        },
        "claim_boundary": (
            "Candidate coverage and matched ranker training improve offline decision "
            "statistics in a privileged simulator-state protocol, but the two-seed "
            "closed-loop balanced-utility intervention does not establish a stable strict "
            "task-success gain. The evidence localizes a remaining distribution-shift and "
            "multi-step closure boundary rather than a universal-world-model result."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_augmented_evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    closed = report["closed_loop_balanced_utility"]["dense_ranker_dense_minus_baseline_ranker_baseline"]
    rank = report["ranker_distribution_matching"]["dense_ranker_minus_baseline_ranker_on_dense"]
    lines = [
        "# Candidate-augmented PushT evidence",
        "",
        "| analysis | estimand | estimate | 95% CI | n |",
        "|---|---|---:|---:|---:|",
        "| Matched ranker training | selected-feasible change on dense fan | %.3f | [%.3f, %.3f] | %d |" % (rank["selected_feasible"]["estimate"], rank["selected_feasible"]["ci95_low"], rank["selected_feasible"]["ci95_high"], rank["selected_feasible"]["n"]),
        "| Closed loop | strict-success change | %.3f | [%.3f, %.3f] | %d |" % (closed["strict_success"]["estimate"], closed["strict_success"]["ci95_low"], closed["strict_success"]["ci95_high"], closed["strict_success"]["n"]),
        "| Closed loop | final XY change (pixels) | %.2f | [%.2f, %.2f] | %d |" % (closed["final_xy"]["estimate"], closed["final_xy"]["ci95_low"], closed["final_xy"]["ci95_high"], closed["final_xy"]["n"]),
        "| Closed loop | final angle change (radians) | %.3f | [%.3f, %.3f] | %d |" % (closed["final_angle"]["estimate"], closed["final_angle"]["ci95_low"], closed["final_angle"]["ci95_high"], closed["final_angle"]["n"]),
        "",
        report["claim_boundary"],
        "",
    ]
    (output / "candidate_augmented_evidence.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
