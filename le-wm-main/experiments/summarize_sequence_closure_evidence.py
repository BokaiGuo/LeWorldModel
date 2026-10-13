"""Aggregate the learned PushT sequence-closure evidence without rerunning data."""

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


def read_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data["sanity_checks"]["runtime_candidate_successor_verification"]:
        raise ValueError(f"runtime verification unexpectedly enabled: {path}")
    if not data["sanity_checks"]["all_finite"]:
        raise ValueError(f"non-finite artifact: {path}")
    return data


def paired_rows(data: dict[str, Any], new: str, old: str, metric: str) -> np.ndarray:
    rows = {item["variant"]: item["rows"] for item in data["summaries"]}
    return np.asarray([a[metric] - b[metric] for a, b in zip(rows[new], rows[old])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/pusht/lewm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    root = Path(args.root)

    closure_paths = [
        root / "learned_sequence_verifier_continuous_local_seed20261089" / "learned_sequence_verifier.json",
        root / "learned_sequence_verifier_continuous_local_seed20261090" / "learned_sequence_verifier.json",
        root / "learned_sequence_verifier_continuous_highcoverage_seed20261091" / "learned_sequence_verifier.json",
        root / "learned_sequence_verifier_continuous_highcoverage_seed20261092" / "learned_sequence_verifier.json",
    ]
    high_paths = closure_paths[2:]
    closure = [read_report(path) for path in closure_paths]
    high = [read_report(path) for path in high_paths]

    variants = ("sequence_soft_contact", "sequence_continuous_rank")
    aggregate: dict[str, Any] = {}
    for variant in variants:
        rows = [row for data in closure for row in next(item["rows"] for item in data["summaries"] if item["variant"] == variant)]
        aggregate[variant] = {
            metric: bootstrap(np.asarray([row[metric] for row in rows]), args.bootstrap, 100 + index)
            for index, metric in enumerate(("strict_success", "final_xy", "final_angle"))
        }

    paired_high: dict[str, Any] = {}
    for index, metric in enumerate(("strict_success", "final_xy", "final_angle")):
        values = np.concatenate([
            paired_rows(data, "sequence_continuous_rank", "sequence_soft_contact", metric)
            for data in high
        ])
        paired_high[metric] = bootstrap(values, args.bootstrap, 200 + index)
        paired_high[metric]["raw_sum_new"] = int(sum(
            row[metric] for data in high
            for row in next(item["rows"] for item in data["summaries"] if item["variant"] == "sequence_continuous_rank")
        )) if metric == "strict_success" else None
        paired_high[metric]["raw_sum_old"] = int(sum(
            row[metric] for data in high
            for row in next(item["rows"] for item in data["summaries"] if item["variant"] == "sequence_soft_contact")
        )) if metric == "strict_success" else None

    ranking = {}
    for seed in ("20261091", "20261092"):
        ranking[str(seed)] = read_report(root / f"learned_sequence_ranking_audit_seed{seed}" / "learned_sequence_ranking_audit.json")
    ranking_paired = {
        metric: bootstrap(
            np.concatenate([
                np.asarray([row[metric] for row in data["summaries"]["sequence_continuous_rank"]["rows"]])
                - np.asarray([row[metric] for row in data["summaries"]["sequence_soft_contact"]["rows"]])
                for data in ranking.values()
            ]),
            args.bootstrap,
            300 + index,
        )
        for index, metric in enumerate((
            "selected_feasible", "top8_hit", "top32_hit", "selected_rank_by_true_score",
            "selected_angle_progress", "selected_xy_progress",
        ))
    }

    report = {
        "protocol": {
            "closure_seed_count": len(closure),
            "high_coverage_seed_count": len(high),
            "ranking_audit_seed_count": len(ranking),
            "all_runtime_candidate_successor_verification": False,
            "source_paths": [str(path) for path in closure_paths],
        },
        "aggregate_closure": aggregate,
        "high_coverage_paired_continuous_minus_binary": paired_high,
        "ranking_paired_continuous_minus_binary": ranking_paired,
        "claim_boundary": (
            "Privileged simulator-state evidence for a learned macro-action "
            "decision interface. Continuous goal-conditioned ranking improves "
            "pose quality and true utility ordering, but does not establish a "
            "universal world model, visual planning, physical latent causality, "
            "or a significant binary task-success gain."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sequence_closure_evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Learned sequence closure evidence", "",
        "| selector | strict success | final XY | final angle |", "|---|---:|---:|---:|",
    ]
    for variant in variants:
        m = aggregate[variant]
        lines.append("| %s | %.3f [%.3f, %.3f] | %.2f [%.2f, %.2f] | %.3f [%.3f, %.3f] |" % (
            variant,
            m["strict_success"]["estimate"], m["strict_success"]["ci95_low"], m["strict_success"]["ci95_high"],
            m["final_xy"]["estimate"], m["final_xy"]["ci95_low"], m["final_xy"]["ci95_high"],
            m["final_angle"]["estimate"], m["final_angle"]["ci95_low"], m["final_angle"]["ci95_high"],
        ))
    lines += ["", "High-coverage paired continuous minus binary:", ""]
    for metric, value in paired_high.items():
        lines.append("- `%s`: %.6f [%.6f, %.6f]" % (metric, value["estimate"], value["ci95_low"], value["ci95_high"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "sequence_closure_evidence.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
