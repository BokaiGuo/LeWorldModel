"""Aggregate RGB-compatible indexed-canonical task selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_rgb_indexed_canonical_task_selection import audit


def _two_level(case_values: dict[str, np.ndarray], samples: int, seed: int) -> dict[str, Any]:
    names = list(case_values)
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean(case_values[name]) for name in names]))
    draws = []
    for _ in range(samples):
        selected = rng.integers(0, len(names), size=len(names))
        means = []
        for index in selected:
            values = case_values[names[index]]
            roots = rng.integers(0, len(values), size=len(values))
            means.append(float(np.mean(values[roots])))
        draws.append(float(np.mean(means)))
    return {
        "case_count": len(names),
        "point": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def aggregate(cases: list[tuple[str, Path, Path, Path, Path]], output: Path, samples: int, seed: int) -> dict[str, Any]:
    reports = {}
    case_summary = {}
    metrics: dict[str, dict[str, np.ndarray]] = {"regret_reduction": {}, "consistency_gain": {}}
    for name, dataset, train_report, calibration_report, outcome_report in cases:
        print(f"[rgb-indexed-task-panel] auditing {name}")
        report = audit(dataset, train_report, calibration_report, outcome_report)
        baseline = report["results"]["rgb_source_plus_delta"]
        candidate = report["results"]["rgb_indexed_canonical_action"]
        base_roots = baseline["root_metrics"]["outcome"]
        candidate_roots = candidate["root_metrics"]["outcome"]
        roots = sorted(set(base_roots) & set(candidate_roots))
        regret = np.asarray([base_roots[root]["regret"] - candidate_roots[root]["regret"] for root in roots], dtype=np.float64)
        consistency = np.asarray([
            candidate_roots[root]["execution_consistency"] - base_roots[root]["execution_consistency"]
            for root in roots
            if base_roots[root]["execution_consistency"] is not None and candidate_roots[root]["execution_consistency"] is not None
        ], dtype=np.float64)
        metrics["regret_reduction"][name] = regret
        metrics["consistency_gain"][name] = consistency
        case_summary[name] = {
            "root_count": len(roots),
            "regret_reduction": float(np.mean(regret)),
            "consistency_gain": float(np.mean(consistency)),
            "baseline_outcome_regret": baseline["candidate_regret"]["outcome"]["mean_regret"],
            "candidate_outcome_regret": candidate["candidate_regret"]["outcome"]["mean_regret"],
            "baseline_outcome_consistency": baseline["execution_consistency"]["outcome"]["pairs"]["execution_consistency"],
            "candidate_outcome_consistency": candidate["execution_consistency"]["outcome"]["pairs"]["execution_consistency"],
        }
        reports[name] = report
    aggregate_result = {}
    for index, metric in enumerate(("regret_reduction", "consistency_gain")):
        aggregate_result[metric] = {
            "two_level_case_and_root_bootstrap": _two_level(metrics[metric], samples, seed + index),
            "all_case_points_positive": bool(all(np.mean(values) > 0.0 for values in metrics[metric].values())),
        }
    result = {
        "schema": "gdd_wm_rgb_indexed_canonical_task_panel.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "case_unit": "architecture_by_seed_checkpoint",
            "fit_split": "train",
            "evaluation_split": "outcome",
            "baseline": "rgb_source_plus_delta",
            "candidate": "rgb_indexed_canonical_action",
            "outcome_used_for_fit_or_variant_selection": False,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        },
        "case_order": [name for name, *_ in cases],
        "case_summary": case_summary,
        "aggregate": aggregate_result,
        "reports": reports,
        "claim_boundary": (
            "offline RGB-compatible task readout with train-derived indexed canonical actions and simulator labels; "
            "not closed-loop RGB planning, D-JEPA preference validation, or physical causality"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="name|candidate_dataset|train_rgb|calibration_rgb|outcome_rgb")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    cases = []
    for specification in args.case:
        parts = specification.split("|")
        if len(parts) != 5 or not all(parts):
            raise ValueError("each case must be name|candidate_dataset|train_rgb|calibration_rgb|outcome_rgb")
        cases.append((parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3]), Path(parts[4])))
    report = aggregate(cases, args.output, args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps({"case_summary": report["case_summary"], "aggregate": report["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
