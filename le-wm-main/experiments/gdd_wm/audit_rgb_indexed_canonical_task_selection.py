"""Audit RGB-compatible task selection with indexed canonical actions.

This joins frozen RGB successor reports to one immutable candidate bank.  The
task readout is fit on train-root simulator cost labels only.  Calibration is
reported but never used to change the feature family; outcome roots measure
the independent candidate-ranking consequence.

No test-root simulator state is used to construct features.  The only action
geometry is a train-derived lookup keyed by the shared candidate id.  The
result is still an offline RGB-compatible task-readout audit, not closed-loop
RGB planning or a causal-dynamics result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.audit_indexed_canonical_action_geometry import (
    _canonical_summary,
    _fit_indexed_bank,
)
from experiments.gdd_wm.audit_transfer_sufficient_feature_families import (
    _evaluate,
    _fit_ridge,
    _predict,
)


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    gates = report.get("gates", {})
    required = ("renderer_consistency", "realized_rgb_latents_available", "pca_projected_successor_comparison")
    if not all(bool(gates.get(name)) for name in required):
        raise ValueError(f"RGB successor gates are not all passed: {path}")
    return report


def _join_rgb_reports(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    joined: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for row in _load_report(path).get("rows", []):
            key = (str(row["instance_id"]), int(row["candidate_id"]))
            if key in joined:
                raise ValueError(f"duplicate RGB successor row {key}")
            joined[key] = row
    return joined


def _features(row: dict[str, Any], rgb: dict[str, Any], bank: dict[int, np.ndarray], family: str) -> np.ndarray:
    source = np.asarray(rgb["source_projection"], dtype=np.float64)
    predicted = np.asarray(rgb["predicted_terminal_projection"], dtype=np.float64)
    if family == "rgb_source_plus_delta":
        return np.concatenate([source, predicted - source])
    if family == "rgb_indexed_canonical_action":
        candidate = int(row["candidate_id"])
        if candidate not in bank:
            raise ValueError(f"candidate {candidate} missing from train-derived action bank")
        return np.concatenate([source, predicted - source, bank[candidate]])
    raise ValueError(f"unknown RGB feature family {family!r}")


def audit(
    candidate_dataset: Path,
    train_rgb_report: Path,
    calibration_rgb_report: Path,
    outcome_rgb_report: Path,
    *,
    alpha: float = 1.0e-2,
    margin: float = 1.0e-3,
) -> dict[str, Any]:
    dataset = json.loads(candidate_dataset.read_text(encoding="utf-8"))
    rows = [dict(row) for row in dataset.get("records", [])]
    if not rows:
        raise ValueError("candidate dataset has no records")
    rgb_rows = _join_rgb_reports([train_rgb_report, calibration_rgb_report, outcome_rgb_report])
    for row in rows:
        key = (str(row["instance_id"]), int(row["candidate_id"]))
        if key not in rgb_rows:
            raise ValueError(f"missing RGB successor row {key}")
        if str(rgb_rows[key]["split"]) != str(row["split"]):
            raise ValueError(f"split mismatch for {key}")
    train = [row for row in rows if str(row["split"]) == "train"]
    calibration = [row for row in rows if str(row["split"]) == "calibration"]
    outcome = [row for row in rows if str(row["split"]) == "outcome"]
    if not train or not calibration or not outcome:
        raise ValueError("train, calibration, and outcome rows are required")
    train_roots = {str(row["instance_id"]) for row in train}
    if train_roots & ({str(row["instance_id"]) for row in calibration} | {str(row["instance_id"]) for row in outcome}):
        raise ValueError("RGB task selection requires root-disjoint splits")
    bank = _fit_indexed_bank(train)
    targets = np.asarray([float(row["real_task_cost"]) for row in rows], dtype=np.float64)
    train_mask = np.asarray([str(row["split"]) == "train" for row in rows])
    results = {}
    for family in ("rgb_source_plus_delta", "rgb_indexed_canonical_action"):
        features = np.asarray([_features(row, rgb_rows[(str(row["instance_id"]), int(row["candidate_id"]))], bank, family) for row in rows], dtype=np.float64)
        fit = _fit_ridge(features[train_mask], targets[train_mask], alpha)
        scores = _predict(features, fit)
        results[family] = {
            "feature_dimension": int(features.shape[1]),
            "train_cost_fit_rmse": float(np.sqrt(np.mean((_predict(features[train_mask], fit) - targets[train_mask]) ** 2))),
            **_evaluate(rows, scores, margin),
        }
    return {
        "schema": "gdd_wm_rgb_indexed_canonical_task_selection.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "train_rgb_report": str(train_rgb_report.resolve()),
            "calibration_rgb_report": str(calibration_rgb_report.resolve()),
            "outcome_rgb_report": str(outcome_rgb_report.resolve()),
            "rgb_feature_fit_split": "train",
            "action_bank_fit_split": "train",
            "evaluation_splits": ["calibration", "outcome"],
            "test_action_features": "train_derived_candidate_id_lookup",
            "outcome_used_for_fit_or_feature_selection": False,
            "ridge_alpha": float(alpha),
            "pair_margin": float(margin),
        },
        "bank": {
            "candidate_count": len(bank),
            "candidate_ids": sorted(bank),
            "train_derived_shared_bank": True,
            "all_rows_canonical_deviation": float(max(
                np.linalg.norm(_canonical_summary(row) - bank[int(row["candidate_id"])])
                for row in rows
            )),
        },
        "results": results,
        "claim_boundary": (
            "offline RGB-compatible task readout with train-derived indexed canonical actions and simulator labels; "
            "not closed-loop RGB planning, D-JEPA preference validation, or physical causality"
        ),
        "input_metadata": dataset.get("metadata", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", required=True, type=Path)
    parser.add_argument("--train-rgb-report", required=True, type=Path)
    parser.add_argument("--calibration-rgb-report", required=True, type=Path)
    parser.add_argument("--outcome-rgb-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    args = parser.parse_args()
    report = audit(
        args.candidate_dataset,
        args.train_rgb_report,
        args.calibration_rgb_report,
        args.outcome_rgb_report,
        alpha=args.alpha,
        margin=args.margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
