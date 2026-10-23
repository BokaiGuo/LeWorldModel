"""Summarize sealed PointMaze temporal-memory panels without mixing regimes.

This utility aggregates only root-level paired regrets from already written
score reports.  It deliberately refuses to pool numeric action-OOD panels with
the shared-candidate in-support mechanism control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"recurrent", "action_only", "outcome_root_ids", "active_policy"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError(f"{path}: not a V6 score report")
    return value


def _paired_values(report: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    recurrent, baseline = report["recurrent"]["roots"], report["action_only"]["roots"]
    roots = sorted(recurrent)
    if roots != sorted(baseline) or roots != sorted(report["outcome_root_ids"]):
        raise ValueError("score report has inconsistent root bindings")
    values = np.asarray([float(baseline[root]["regret"]) - float(recurrent[root]["regret"]) for root in roots])
    return roots, values


def _bootstrap(values: np.ndarray, *, seed: int, samples: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("requires non-empty root-level gains")
    generator = np.random.default_rng(seed)
    draws = np.asarray([generator.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return {
        "mean_gain": float(values.mean()),
        "lower": float(np.quantile(draws, .025)),
        "upper": float(np.quantile(draws, .975)),
        "root_count": int(len(values)),
    }


def summarize(report_paths: list[Path], *, regime: str, seed: int = 2026102643, samples: int = 10000) -> dict[str, Any]:
    if regime not in {"numeric_ood", "shared_candidate_in_support"}:
        raise ValueError("regime must be numeric_ood or shared_candidate_in_support")
    panels, all_roots, all_values = [], set(), []
    for path in report_paths:
        report = _read(path)
        roots, values = _paired_values(report)
        duplicated = all_roots & set(roots)
        if duplicated:
            raise ValueError(f"outcome roots overlap across panels: {sorted(duplicated)}")
        all_roots.update(roots); all_values.extend(values.tolist())
        panels.append({
            "path": str(path.resolve()),
            "active_policy": report["active_policy"],
            "paired_recurrent_over_action_only": _bootstrap(values, seed=seed + len(panels), samples=samples),
            "root_count": len(roots),
        })
    aggregate = _bootstrap(np.asarray(all_values, dtype=np.float64), seed=seed, samples=samples)
    return {
        "schema": "pointmaze_temporal_ood_v6_summary.v1",
        "candidate_generalization_regime": regime,
        "panels": panels,
        "pooled_root_level_paired_recurrent_over_action_only": aggregate,
        "outcome_root_count": len(all_roots),
        "claim_boundary": "Aggregation preserves the declared candidate-support regime. It is offline renderer-backed PointMaze evidence, not live visual planning.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--regime", choices=("numeric_ood", "shared_candidate_in_support"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    result = summarize(args.reports, regime=args.regime, samples=args.bootstrap_samples)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
