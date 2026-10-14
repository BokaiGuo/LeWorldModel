"""Aggregate paired relative-effect audits across learned-model cases.

The input panel contains one frozen learned checkpoint per case.  This
aggregator keeps the experimental unit explicit: outcome roots are paired
within a case, while architecture/seed cases are a second uncertainty layer.
It therefore does not treat all root-by-case cells as independent evidence
for cross-architecture stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "point": None, "ci95": [None, None]}
    draws = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap = np.mean(values[draws], axis=1)
    return {
        "count": int(values.size),
        "point": float(np.mean(values)),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
    }


def _case_effects(
    panel: dict[str, Any],
    case_names: list[str],
    baseline: str,
    variant: str,
) -> dict[str, dict[str, np.ndarray]]:
    cases: dict[str, dict[str, np.ndarray]] = {}
    for name in case_names:
        variants = panel["panel"][name]["variants"]
        if baseline not in variants or variant not in variants:
            raise ValueError(f"case {name!r} is missing baseline or variant")
        base = variants[baseline]["root_metrics"].get("outcome", {})
        candidate = variants[variant]["root_metrics"].get("outcome", {})
        roots = sorted(set(base) & set(candidate))
        consistency = []
        regret = []
        for root in roots:
            base_cons = base[root].get("execution_consistency")
            variant_cons = candidate[root].get("execution_consistency")
            base_regret = base[root].get("regret")
            variant_regret = candidate[root].get("regret")
            if base_cons is not None and variant_cons is not None:
                consistency.append(float(variant_cons) - float(base_cons))
            if base_regret is not None and variant_regret is not None:
                regret.append(float(base_regret) - float(variant_regret))
        cases[name] = {
            "consistency_gain": np.asarray(consistency, dtype=np.float64),
            "regret_reduction": np.asarray(regret, dtype=np.float64),
        }
    return cases


def aggregate(
    panel_path: Path,
    output_path: Path,
    *,
    baseline: str,
    variant: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    case_names = list(payload["panel_member_order"])
    cases = _case_effects(payload, case_names, baseline, variant)
    rng = np.random.default_rng(seed)

    case_summary: dict[str, Any] = {}
    for name, effects in cases.items():
        case_summary[name] = {
            metric: {
                "root_count": int(values.size),
                "point": None if values.size == 0 else float(np.mean(values)),
                "min": None if values.size == 0 else float(np.min(values)),
                "max": None if values.size == 0 else float(np.max(values)),
            }
            for metric, values in effects.items()
        }

    def case_means(metric: str) -> np.ndarray:
        return np.asarray(
            [np.mean(cases[name][metric]) for name in case_names if cases[name][metric].size],
            dtype=np.float64,
        )

    def two_level(metric: str) -> np.ndarray:
        """Resample cases and then roots within each sampled case."""
        draws = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sampled_cases = rng.integers(0, len(case_names), size=len(case_names))
            case_values = []
            for case_index in sampled_cases:
                values = cases[case_names[case_index]][metric]
                if values.size:
                    root_indices = rng.integers(0, values.size, size=values.size)
                    case_values.append(float(np.mean(values[root_indices])))
            draws[index] = float(np.mean(case_values)) if case_values else np.nan
        return draws[np.isfinite(draws)]

    aggregate_report: dict[str, Any] = {}
    for metric in ("consistency_gain", "regret_reduction"):
        pooled = np.concatenate([cases[name][metric] for name in case_names])
        two_level_draws = two_level(metric)
        original_case_means = case_means(metric)
        aggregate_report[metric] = {
            "case_macro_mean": _ci(original_case_means, samples, rng),
            "root_cell_mean": _ci(pooled, samples, rng),
            "two_level_case_and_root_bootstrap": {
                "count": int(two_level_draws.size),
                "point": None
                if original_case_means.size == 0
                else float(np.mean(original_case_means)),
                "ci95": [
                    None
                    if two_level_draws.size == 0
                    else float(np.quantile(two_level_draws, 0.025)),
                    None
                    if two_level_draws.size == 0
                    else float(np.quantile(two_level_draws, 0.975)),
                ],
            },
            "all_case_points_positive": bool(all(np.mean(cases[name][metric]) > 0 for name in case_names)),
        }

    report = {
        "schema": "gdd_wm_relative_action_effect_aggregate.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "source_panel": str(panel_path.resolve()),
            "baseline": baseline,
            "variant": variant,
            "outcome_used_for_fit_or_variant_selection": False,
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "case_unit": "architecture_by_seed_checkpoint",
            "root_unit": "episode_root",
            "uncertainty_note": (
                "case_macro and two-level bootstrap preserve the distinction between "
                "root uncertainty and architecture/seed stability"
            ),
        },
        "case_order": case_names,
        "case_summary": case_summary,
        "aggregate": aggregate_report,
        "claim_boundary": (
            "offline canonical goal-frame action-effect diagnostic across two architectures "
            "and two seeds; not D-JEPA preference validation, RGB online planning, causal "
            "dynamics, or universal world-model evidence"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="root_centered_source_delta_action")
    parser.add_argument("--variant", default="root_centered_source_delta_goalframe_action")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261014)
    args = parser.parse_args()
    report = aggregate(
        args.panel,
        args.output,
        baseline=args.baseline,
        variant=args.variant,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
