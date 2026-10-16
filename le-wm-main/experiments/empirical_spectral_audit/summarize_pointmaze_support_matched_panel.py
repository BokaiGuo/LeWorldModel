"""Summarize and pair PointMaze support-matched external panels.

The primary comparison is frozen by ``architecture x seed x stage``.  The
script never selects a checkpoint using external outcomes; it reports every
matched cell and bootstraps the paired chart difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _report_key(report: dict[str, Any]) -> tuple[str, int, int]:
    checkpoint = Path(str(report["checkpoint"]))
    return checkpoint.parent.parent.name, int(checkpoint.parent.name.removeprefix("seed_")), int(
        checkpoint.stem.removeprefix("stage_")
    )


def _load_panel(manifest_path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pointmaze_support_matched_external_panel.v2":
        raise ValueError(f"unexpected panel schema in {manifest_path}")
    reports = {}
    for report_path in manifest.get("reports", []):
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if report.get("protocol", {}).get("device") != "cuda":
            raise ValueError(f"non-CUDA report in {manifest_path}: {report_path}")
        key = _report_key(report)
        if key in reports:
            raise ValueError(f"duplicate report key {key} in {manifest_path}")
        reports[key] = report
    return reports


def _bootstrap(values: np.ndarray, seed: int, samples: int = 10000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "by_seed_mean": {
            str(seed): float(np.mean([float(row[metric]) for row in rows if int(row["seed"]) == seed]))
            for seed in sorted({int(row["seed"]) for row in rows})
        },
    }


def _flat(report: dict[str, Any]) -> dict[str, Any]:
    key = _report_key(report)
    return {
        "architecture": key[0],
        "seed": key[1],
        "stage": key[2],
        "action_inverse_risk": float(report["action_risk"]["action_inverse_risk"]),
        "planning_risk": float(report["planning_consequence"]["planning_risk"]),
        "paired_roots": int(report["action_risk"]["paired_roots"]),
    }


def summarize(
    goal_manifest: Path,
    output: Path,
    *,
    raw_manifest: Path | None = None,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    goal = _load_panel(goal_manifest)
    goal_rows = [_flat(report) for _, report in sorted(goal.items())]
    report: dict[str, Any] = {
        "schema": "pointmaze_support_matched_chart_comparison.v1",
        "status": "completed_diagnostic",
        "protocol": {
            "goal_relative_panel": str(goal_manifest.resolve()),
            "raw_panel": None if raw_manifest is None else str(raw_manifest.resolve()),
            "pair_unit": "architecture_by_seed_by_training_stage",
            "device": "cuda",
            "external_outcome_used_for_checkpoint_selection": False,
            "bootstrap_samples": int(bootstrap_samples),
        },
        "goal_relative": {
            "report_count": len(goal_rows),
            "action_inverse_risk": _metric_summary(goal_rows, "action_inverse_risk"),
            "planning_risk": _metric_summary(goal_rows, "planning_risk"),
        },
        "claim_boundary": (
            "support-matched PointMaze external action-transfer diagnostic; the planning field is an offline "
            "horizon consequence, not closed-loop RGB planning or universal world-model evidence"
        ),
    }
    if raw_manifest is not None:
        raw = _load_panel(raw_manifest)
        common = sorted(set(goal) & set(raw))
        if not common:
            raise ValueError("goal-relative and raw panels have no matched architecture/seed/stage cells")
        pairs = []
        for key in common:
            goal_row = _flat(goal[key])
            raw_row = _flat(raw[key])
            pairs.append({
                "architecture": key[0],
                "seed": key[1],
                "stage": key[2],
                "raw_action_inverse_risk": raw_row["action_inverse_risk"],
                "goal_relative_action_inverse_risk": goal_row["action_inverse_risk"],
                "action_risk_reduction": raw_row["action_inverse_risk"] - goal_row["action_inverse_risk"],
                "raw_planning_risk": raw_row["planning_risk"],
                "goal_relative_planning_risk": goal_row["planning_risk"],
                "planning_risk_reduction": raw_row["planning_risk"] - goal_row["planning_risk"],
            })
        report["matched_comparison"] = {
            "matched_cell_count": len(pairs),
            "all_cells_action_risk_improve": bool(all(row["action_risk_reduction"] > 0 for row in pairs)),
            "action_risk_reduction": _bootstrap(
                np.asarray([row["action_risk_reduction"] for row in pairs]), 20261016, bootstrap_samples
            ),
            "planning_risk_reduction": _bootstrap(
                np.asarray([row["planning_risk_reduction"] for row in pairs]), 20261017, bootstrap_samples
            ),
            "by_architecture": {
                architecture: {
                    "cells": sum(row["architecture"] == architecture for row in pairs),
                    "action_risk_reduction": _bootstrap(
                        np.asarray([row["action_risk_reduction"] for row in pairs if row["architecture"] == architecture]),
                        20261018,
                        bootstrap_samples,
                    ),
                    "planning_risk_reduction": _bootstrap(
                        np.asarray([row["planning_risk_reduction"] for row in pairs if row["architecture"] == architecture]),
                        20261019,
                        bootstrap_samples,
                    ),
                }
                for architecture in sorted({row["architecture"] for row in pairs})
            },
            "pairs": pairs,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-relative-manifest", required=True, type=Path)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = summarize(
        args.goal_relative_manifest,
        args.output,
        raw_manifest=args.raw_manifest,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({key: report[key] for key in ("goal_relative", "matched_comparison") if key in report}, indent=2))


if __name__ == "__main__":
    main()
