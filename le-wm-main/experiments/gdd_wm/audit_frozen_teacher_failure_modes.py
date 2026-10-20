"""Stratified failure-mode audit for a frozen predictive teacher.

This audit is deliberately read-only.  It does not fit a teacher, select a
threshold, or use the report to promote a planner.  Its purpose is to
distinguish a global teacher-capacity failure from a root-local action-effect
failure before a new teacher is trained.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty records list")
    return rows


def _group_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = consistent = 0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            teacher_delta = float(left["teacher_score"]) - float(right["teacher_score"])
            cost_delta = float(left["real_task_cost"]) - float(right["real_task_cost"])
            if abs(teacher_delta) <= 1.0e-3 or abs(cost_delta) <= 1.0e-3:
                continue
            pairs += 1
            # Higher teacher score means lower predicted cost.
            consistent += int((teacher_delta > 0.0) == (cost_delta < 0.0))
    return {
        "pair_count": int(pairs),
        "consistent_pair_count": int(consistent),
        "execution_consistency": None if pairs == 0 else float(consistent / pairs),
    }


def _stratify(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[str(key_fn(row))].append(row)
    by_set: dict[str, dict[str, Any]] = {}
    for cell, cell_rows in cells.items():
        sets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            sets[str(row["teacher_set_id"])].append(row)
        cell_pairs = cell_consistent = 0
        set_reports = {}
        for set_id, set_rows in sorted(sets.items()):
            report = _group_consistency(set_rows)
            set_reports[set_id] = report
            cell_pairs += report["pair_count"]
            cell_consistent += report["consistent_pair_count"]
        by_set[cell] = {
            "row_count": len(cell_rows),
            "candidate_set_count": len(sets),
            "pair_count": int(cell_pairs),
            "consistent_pair_count": int(cell_consistent),
            "execution_consistency": None if cell_pairs == 0 else float(cell_consistent / cell_pairs),
            "sets": set_reports,
        }
    return by_set


def _within_teacher_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pair statistics without comparing different decision cells."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["teacher_set_id"])].append(row)
    reports = [_group_consistency(group) for group in groups.values()]
    pairs = sum(item["pair_count"] for item in reports)
    consistent = sum(item["consistent_pair_count"] for item in reports)
    return {
        "pair_count": int(pairs),
        "consistent_pair_count": int(consistent),
        "execution_consistency": None if pairs == 0 else float(consistent / pairs),
        "candidate_set_count": len(groups),
    }


def audit(path: Path) -> dict[str, Any]:
    rows = _load(path)
    error = np.asarray(
        [float(row["teacher_score"]) + float(row["real_task_cost"]) for row in rows],
        dtype=np.float64,
    )
    support_width = np.asarray(
        [float(row["action_support"]["max"]) - float(row["action_support"]["min"]) for row in rows],
        dtype=np.float64,
    )
    score_cost_corr = float(np.corrcoef(-np.asarray([float(r["teacher_score"]) for r in rows]),
                                        np.asarray([float(r["real_task_cost"]) for r in rows]))[0, 1])
    abs_error_support_corr = float(np.corrcoef(np.abs(error), support_width)[0, 1])
    global_report = _within_teacher_sets(rows)
    return {
        "schema": "gdd_wm_frozen_teacher_failure_modes.v0.1",
        "status": "completed_read_only_audit",
        "protocol": {
            "dataset": str(path.resolve()),
            "outcome_used_for_fit_or_selection": False,
            "teacher_refit": False,
            "planner_authorization": False,
        },
        "global": global_report,
        "strata": {
            "split": _stratify(rows, lambda row: str(row.get("split", "missing"))),
            "context": _stratify(rows, lambda row: str(row.get("candidate_bank_context_id", "missing"))),
            "horizon": _stratify(rows, lambda row: str(row.get("predicted_horizon", "missing"))),
            "root": _stratify(rows, lambda row: str(row.get("start_episode_id", row.get("instance_id", "missing")))),
        },
        "support_diagnostics": {
            "teacher_cost_correlation": score_cost_corr,
            "absolute_teacher_residual_support_width_correlation": abs_error_support_corr,
            "support_width_min": float(np.min(support_width)),
            "support_width_max": float(np.max(support_width)),
            "support_width_mean": float(np.mean(support_width)),
            "interpretation": (
                "Root-stratified action-effect ranking is the release bottleneck when "
                "global consistency is moderate but root consistency is heterogeneous."
            ),
        },
        "claim_boundary": (
            "Failure-mode diagnosis only; no teacher correctness, distillation gain, "
            "closed-loop planning, or universal-transfer claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"global": report["global"], "support_diagnostics": report["support_diagnostics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
