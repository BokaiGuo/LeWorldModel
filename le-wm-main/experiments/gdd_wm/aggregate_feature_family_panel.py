"""Aggregate paired candidate-selection consequences across learned cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap = np.mean(values[draws], axis=1)
    return {
        "count": int(len(values)),
        "point": float(np.mean(values)),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def _two_level(
    cases: dict[str, dict[str, np.ndarray]],
    metric: str,
    case_names: list[str],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    point = float(np.mean([np.mean(cases[name][metric]) for name in case_names]))
    draws = []
    for _ in range(samples):
        sampled_cases = rng.integers(0, len(case_names), size=len(case_names))
        means = []
        for index in sampled_cases:
            values = cases[case_names[index]][metric]
            roots = rng.integers(0, len(values), size=len(values))
            means.append(float(np.mean(values[roots])))
        draws.append(float(np.mean(means)))
    return {
        "count": int(len(draws)),
        "point": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def aggregate(
    panel_path: Path,
    output_path: Path,
    *,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    case_names = list(payload["panel_member_order"])
    families = list(payload["panel"][case_names[0]]["feature_families"])
    if baseline not in families:
        raise ValueError(f"missing baseline family {baseline!r}")
    rng = np.random.default_rng(seed)
    report_families = {}
    case_summary = {}
    for family in families:
        case_metrics: dict[str, dict[str, np.ndarray]] = {}
        case_summary[family] = {}
        for name in case_names:
            base = payload["panel"][name]["feature_families"][baseline]["root_metrics"]["outcome"]
            candidate = payload["panel"][name]["feature_families"][family]["root_metrics"]["outcome"]
            roots = sorted(set(base) & set(candidate))
            regret = np.asarray(
                [float(base[root]["regret"]) - float(candidate[root]["regret"]) for root in roots],
                dtype=np.float64,
            )
            consistency = np.asarray(
                [
                    float(candidate[root]["execution_consistency"])
                    - float(base[root]["execution_consistency"])
                    for root in roots
                    if base[root]["execution_consistency"] is not None
                    and candidate[root]["execution_consistency"] is not None
                ],
                dtype=np.float64,
            )
            case_metrics[name] = {"regret_reduction": regret, "consistency_gain": consistency}
            case_summary[family][name] = {
                "root_count": int(regret.size),
                "regret_reduction": float(np.mean(regret)),
                "consistency_gain": float(np.mean(consistency)),
            }
        report_families[family] = {}
        for metric in ("regret_reduction", "consistency_gain"):
            pooled = np.concatenate([case_metrics[name][metric] for name in case_names])
            case_means = np.asarray([np.mean(case_metrics[name][metric]) for name in case_names])
            report_families[family][metric] = {
                "case_macro_mean": _ci(case_means, samples, rng),
                "root_cell_mean": _ci(pooled, samples, rng),
                "two_level_case_and_root_bootstrap": _two_level(
                    case_metrics, metric, case_names, samples, rng
                ),
                "all_case_points_positive": bool(
                    all(np.mean(case_metrics[name][metric]) > 0 for name in case_names)
                ),
            }
    report = {
        "schema": "gdd_wm_feature_family_consequence_aggregate.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "source_panel": str(panel_path.resolve()),
            "baseline": baseline,
            "fit_split": "train",
            "evaluation_split": "outcome",
            "outcome_used_for_fit_or_family_selection": False,
            "case_unit": "architecture_by_seed_checkpoint",
            "root_unit": "outcome_episode_root",
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        },
        "case_order": case_names,
        "case_summary": case_summary,
        "families": report_families,
        "claim_boundary": (
            "train-only RGB-compatible candidate-selection consequence on simulator-labeled "
            "outcome roots; not D-JEPA teacher validation, closed-loop RGB planning, or "
            "universal world-model evidence"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="source_plus_delta")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    report = aggregate(
        args.panel,
        args.output,
        baseline=args.baseline,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report["families"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
