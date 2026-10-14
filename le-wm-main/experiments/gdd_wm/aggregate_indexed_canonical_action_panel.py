"""Aggregate train-derived indexed canonical-action consequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_indexed_canonical_action_geometry import audit
from experiments.gdd_wm.audit_transfer_sufficient_feature_families import (
    audit as audit_feature_families,
)


def _two_level(case_values: dict[str, np.ndarray], samples: int, seed: int) -> dict[str, Any]:
    names = list(case_values)
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean(case_values[name]) for name in names]))
    draws = []
    for _ in range(samples):
        sampled = rng.integers(0, len(names), size=len(names))
        means = []
        for index in sampled:
            values = case_values[names[index]]
            roots = rng.integers(0, len(values), size=len(values))
            means.append(float(np.mean(values[roots])))
        draws.append(float(np.mean(means)))
    return {
        "case_count": len(names),
        "point": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def aggregate(cases: list[tuple[str, Path, Path]], output: Path, samples: int, seed: int) -> dict[str, Any]:
    reports = {}
    case_metrics: dict[str, dict[str, np.ndarray]] = {}
    case_summary = {}
    for name, dataset, arrays in cases:
        print(f"[indexed-canonical-panel] auditing {name}")
        candidate = audit(dataset, arrays)
        baseline = audit_feature_families(dataset, arrays)
        candidate_roots = candidate["task_readout"]["root_metrics"]["outcome"]
        baseline_roots = baseline["feature_families"]["source_plus_delta"]["root_metrics"]["outcome"]
        roots = sorted(set(candidate_roots) & set(baseline_roots))
        regret = np.asarray(
            [float(baseline_roots[root]["regret"]) - float(candidate_roots[root]["regret"]) for root in roots],
            dtype=np.float64,
        )
        consistency = np.asarray(
            [
                float(candidate_roots[root]["execution_consistency"])
                - float(baseline_roots[root]["execution_consistency"])
                for root in roots
                if baseline_roots[root]["execution_consistency"] is not None
                and candidate_roots[root]["execution_consistency"] is not None
            ],
            dtype=np.float64,
        )
        case_metrics[name] = {"regret_reduction": regret, "consistency_gain": consistency}
        case_summary[name] = {
            "root_count": len(roots),
            "regret_reduction": float(np.mean(regret)),
            "consistency_gain": float(np.mean(consistency)),
            "candidate_outcome_regret": candidate["task_readout"]["candidate_regret"]["outcome"]["mean_regret"],
            "baseline_outcome_regret": baseline["feature_families"]["source_plus_delta"]["candidate_regret"]["outcome"]["mean_regret"],
            "candidate_outcome_consistency": candidate["task_readout"]["execution_consistency"]["outcome"]["pairs"]["execution_consistency"],
            "baseline_outcome_consistency": baseline["feature_families"]["source_plus_delta"]["execution_consistency"]["outcome"]["pairs"]["execution_consistency"],
        }
        reports[name] = {
            "candidate": candidate,
            "baseline": baseline["feature_families"]["source_plus_delta"],
        }
    result = {
        "schema": "gdd_wm_indexed_canonical_action_panel.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "case_unit": "architecture_by_seed_checkpoint",
            "fit_split": "train",
            "evaluation_split": "outcome",
            "candidate_features": "train_derived_candidate_id_lookup",
            "baseline": "source_plus_delta",
            "outcome_used_for_fit_or_variant_selection": False,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        },
        "case_order": [name for name, _, _ in cases],
        "case_summary": case_summary,
        "aggregate": {
            metric: {
                "two_level_case_and_root_bootstrap": _two_level(
                    {name: values[metric] for name, values in case_metrics.items()}, samples, seed + index
                ),
                "all_case_points_positive": bool(all(np.mean(values[metric]) > 0.0 for values in case_metrics.values())),
            }
            for index, metric in enumerate(("regret_reduction", "consistency_gain"))
        },
        "reports": reports,
        "claim_boundary": (
            "offline train-derived indexed canonical action geometry with simulator-labeled task readout; "
            "not D-JEPA preference validation, physical causality, or closed-loop RGB planning"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="name|candidate_dataset|arrays")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    cases = []
    for specification in args.case:
        parts = specification.split("|")
        if len(parts) != 3 or not all(parts):
            raise ValueError("each --case must be name|candidate_dataset|arrays")
        cases.append((parts[0], Path(parts[1]), Path(parts[2])))
    report = aggregate(cases, args.output, args.bootstrap_samples, args.bootstrap_seed)
    print(json.dumps({"case_summary": report["case_summary"], "aggregate": report["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
