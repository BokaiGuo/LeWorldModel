"""Summarize the predeclared train-only teacher feature-family panel.

The script reports every family on the same held-out outcome roots and uses
root bootstrap intervals.  It never selects a family or changes a controller;
the action-only family is retained as a confounding control.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected records")
    return list(payload["records"])


def _pair_consistency(rows: list[dict[str, Any]], margin: float) -> tuple[int, int]:
    correct = total = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            teacher_delta = float(left["teacher_score"]) - float(right["teacher_score"])
            real_delta = float(right["real_task_cost"]) - float(left["real_task_cost"])
            if abs(teacher_delta) <= margin or abs(real_delta) <= margin:
                continue
            total += 1
            correct += int((teacher_delta > 0.0) == (real_delta > 0.0))
    return correct, total


def _root_metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, float | int]:
    correct, total = _pair_consistency(rows, margin)
    selected = max(rows, key=lambda row: (float(row["teacher_score"]), str(row["candidate_id"])))
    oracle_cost = min(float(row["real_task_cost"]) for row in rows)
    return {
        "pairwise_consistency": None if total == 0 else correct / total,
        "comparable_pairs": total,
        "regret": float(float(selected["real_task_cost"]) - oracle_cost),
        "selected_success": float(bool(selected.get("real_success", False))),
    }


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def summarize(
    cases: dict[str, Path], *, margin: float = 1.0e-3, draws: int = 10000, seed: int = 20261016,
    baseline: str = "predicted_terminal",
) -> dict[str, Any]:
    root_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for family, path in cases.items():
        rows = [row for row in _load(path) if str(row.get("split")) == "outcome"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["instance_id"])].append(row)
        root_metrics[family] = {
            root: _root_metrics(group, margin)
            for root, group in sorted(grouped.items())
        }
    root_sets = [set(values) for values in root_metrics.values()]
    common_roots = set.intersection(*root_sets) if root_sets else set()
    if not common_roots:
        raise ValueError("feature families have no common outcome roots")

    rng = np.random.default_rng(seed)
    summaries: dict[str, Any] = {}
    for family, values in root_metrics.items():
        roots = sorted(common_roots)
        consistency = np.asarray([float(values[root]["pairwise_consistency"]) for root in roots])
        regrets = np.asarray([float(values[root]["regret"]) for root in roots])
        summaries[family] = {
            "outcome_root_count": len(roots),
            "pairwise_consistency": _bootstrap(consistency, rng, draws),
            "mean_regret": _bootstrap(regrets, rng, draws),
            "zero_regret_root_fraction": float(np.mean(regrets <= margin)),
            "root_metrics": values,
        }

    if baseline not in root_metrics:
        raise ValueError(f"baseline family {baseline!r} is not present")
    paired_deltas: dict[str, Any] = {}
    for family, values in root_metrics.items():
        roots = sorted(common_roots)
        consistency_delta = np.asarray([
            float(values[root]["pairwise_consistency"])
            - float(root_metrics[baseline][root]["pairwise_consistency"])
            for root in roots
        ])
        regret_delta = np.asarray([
            float(values[root]["regret"]) - float(root_metrics[baseline][root]["regret"])
            for root in roots
        ])
        paired_deltas[family] = {
            "vs_predicted_terminal": {
                "pairwise_consistency_delta": _bootstrap(consistency_delta, rng, draws),
                "regret_delta": _bootstrap(regret_delta, rng, draws),
            }
        }

    return {
        "schema": "gdd_wm_train_only_teacher_feature_panel.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "family_order": list(cases),
            "fit_split": "train",
            "evaluation_split": "outcome",
            "root_bootstrap": True,
            "bootstrap_draws": int(draws),
            "seed": int(seed),
            "pair_margin": float(margin),
            "family_selection": "not_performed",
            "paired_delta_baseline": baseline,
        },
        "families": summaries,
        "paired_deltas": paired_deltas,
        "claim_boundary": (
            "predeclared train-only feature-family control; action-only is a confounding baseline; "
            "not D-JEPA teacher validation, D2 authorization, or planning evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="family|candidate_dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--baseline", default="predicted_terminal")
    args = parser.parse_args()
    cases: dict[str, Path] = {}
    for spec in args.case:
        parts = spec.split("|", 1)
        if len(parts) != 2 or not all(parts) or parts[0] in cases:
            raise ValueError("each --case must be a unique family|candidate_dataset")
        cases[parts[0]] = Path(parts[1])
    report = summarize(cases, draws=args.draws, seed=args.seed, baseline=args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
