"""Recompute descriptor-panel statistics without rerunning model queries.

This is intentionally separate from the historical runners. It reads their
immutable row outputs, applies tie-corrected ranks, and emits candidate-level
descriptives plus task-stratified summaries. It never mutates the old result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_action_opportunity_stats import partial_rank_correlation, spearman


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "fixed_probe": DEFAULT_ROOT / "results/pc_action_opportunity_descriptor_dev_20261104_codex/result.json",
    "random_probe": DEFAULT_ROOT / "results/pc_action_opportunity_random_probe_dev_20261104_codex/result.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "completed"]


def _numeric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _optional_numeric(rows: list[dict], key: str) -> np.ndarray | None:
    if not rows or any(key not in row for row in rows):
        return None
    return _numeric(rows, key)


def _candidate_summary(rows: list[dict]) -> dict:
    rows = _completed(rows)
    if len(rows) < 3:
        return {"count": len(rows), "status": "insufficient"}
    a = _numeric(rows, "a_max_improvement")
    b = _numeric(rows, "b_max_improvement")
    best = _numeric(rows, "best_score")
    immediate = _numeric(rows, "immediate_score")
    baseline = _optional_numeric(rows, "baseline_score")
    lengths = np.asarray([15 - 5 * int(row["depth"]) for row in rows], dtype=np.float64)
    controls = np.column_stack((immediate, lengths))
    summary = {
        "count": len(rows),
        "status": "completed",
        "spearman_tie_corrected_a_vs_b_improvement": spearman(a, b),
        "spearman_tie_corrected_a_vs_negative_best_score": spearman(a, -best),
        "spearman_tie_corrected_immediate_vs_negative_best_score": spearman(-immediate, -best),
        "partial_rank_a_vs_negative_best_controls_baseline_immediate_horizon": (
            partial_rank_correlation(
                a,
                -best,
                np.column_stack((baseline, controls)) if baseline is not None else controls,
            )
        ),
        "partial_rank_b_vs_negative_best_controls_baseline_immediate_horizon": (
            partial_rank_correlation(
                b,
                -best,
                np.column_stack((baseline, controls)) if baseline is not None else controls,
            )
        ),
        "unique_a_max_improvement": int(np.unique(a).size),
        "unique_best_score": int(np.unique(best).size),
        "mean_remaining_horizon": float(lengths.mean()),
    }
    summary["baseline_score_available"] = baseline is not None
    summary["partial_controls"] = (
        ["baseline_score", "immediate_score", "remaining_horizon"]
        if baseline is not None
        else ["immediate_score", "remaining_horizon"]
    )
    if baseline is not None:
        summary["mean_baseline_score"] = float(baseline.mean())
    return summary


def _task_summary(rows: list[dict]) -> dict:
    """Aggregate within root/arm before reporting association.

    Candidate rows are not treated as independent tasks. Each root/arm cell
    contributes one mean descriptor and one mean outcome to this diagnostic.
    """
    completed = _completed(rows)
    cells: dict[tuple[int, str], list[dict]] = {}
    for row in completed:
        cells.setdefault((int(row["root"]), str(row["arm"])), []).append(row)
    aggregated = []
    for (root, arm), cell in sorted(cells.items()):
        aggregated.append(
            {
                "root": root,
                "arm": arm,
                "a_max_improvement_mean": float(np.mean([r["a_max_improvement"] for r in cell])),
                "b_max_improvement_mean": float(np.mean([r["b_max_improvement"] for r in cell])),
                "best_score_mean": float(np.mean([r["best_score"] for r in cell])),
                "immediate_score_mean": float(np.mean([r["immediate_score"] for r in cell])),
                "baseline_score_mean": (
                    float(np.mean([r["baseline_score"] for r in cell]))
                    if all("baseline_score" in r for r in cell)
                    else None
                ),
                "candidate_count": len(cell),
            }
        )
    if len(aggregated) < 3:
        return {"cell_count": len(aggregated), "status": "insufficient", "cells": aggregated}
    a = np.asarray([r["a_max_improvement_mean"] for r in aggregated], dtype=np.float64)
    b = np.asarray([r["b_max_improvement_mean"] for r in aggregated], dtype=np.float64)
    best = np.asarray([r["best_score_mean"] for r in aggregated], dtype=np.float64)
    immediate = np.asarray([r["immediate_score_mean"] for r in aggregated], dtype=np.float64)
    baseline_values = [r["baseline_score_mean"] for r in aggregated]
    baseline = (
        np.asarray(baseline_values, dtype=np.float64)
        if all(value is not None for value in baseline_values)
        else None
    )
    controls = np.column_stack((immediate,))
    if baseline is not None:
        controls = np.column_stack((baseline, controls))
    return {
        "cell_count": len(aggregated),
        "status": "completed",
        "spearman_tie_corrected_a_vs_b_improvement": spearman(a, b),
        "spearman_tie_corrected_a_vs_negative_best_score": spearman(a, -best),
        "partial_rank_a_vs_negative_best_controls_baseline_immediate": partial_rank_correlation(
            a, -best, controls
        ),
        "baseline_score_available": baseline is not None,
        "partial_controls": (
            ["baseline_score", "immediate_score"]
            if baseline is not None
            else ["immediate_score"]
        ),
        "cells": aggregated,
    }


def audit(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text())
    rows = payload.get("rows", [])
    result = {
        "schema": "pc_action_opportunity_rank_audit.v1",
        "status": "completed",
        "source": str(input_path),
        "source_sha256": _sha256(input_path),
        "source_schema": payload.get("schema"),
        "not_rerun": True,
        "not_independent_confirmation": True,
        "candidate_level": _candidate_summary(rows),
        "task_cell_level": _task_summary(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-input", type=Path, default=DEFAULT_INPUTS["fixed_probe"])
    parser.add_argument("--random-input", type=Path, default=DEFAULT_INPUTS["random_probe"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema": "pc_action_opportunity_rank_audit_panel.v1",
        "status": "completed",
        "fixed_probe": audit(args.fixed_input, args.output.with_name("fixed_probe_audit.json")),
        "random_probe": audit(args.random_input, args.output.with_name("random_probe_audit.json")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
