"""Summarize adaptive memory artifacts at the seeded-split level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("raw_history", "legacy_first_harmonic", "fixed_fourier", "local_difference", "event_aware", "compact_event_aware", "compact_event_zscore", "spectral_phase")


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one split artifact is required")
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        rows.append({
            "source": str(path),
            "split_seed": report["split"].get("split_seed"),
            "fit_rows": report["support"]["fit_rows"],
            "outcome_rows": report["support"]["outcome_rows"],
            "variants": {
                name: {
                    "relative_rmse_reduction": report["variants"][name]["fit_only_rank_matched"]["relative_rmse_reduction"],
                    "lifted_beats_base": report["variants"][name]["fit_only_rank_matched"]["lifted_beats_base"],
                    "base_rmse": report["variants"][name]["fit_only_rank_matched"]["base_outcome_rmse"],
                    "lifted_rmse": report["variants"][name]["fit_only_rank_matched"]["lifted_outcome_rmse"],
                }
                for name in VARIANTS
            },
            "fit_only_selection": report.get("fit_only_selection", {
                "status": "unavailable",
                "selected_variant": None,
            }),
        })

    def values(name: str, key: str) -> np.ndarray:
        return np.asarray([row["variants"][name][key] for row in rows], dtype=np.float64)

    aggregate = {}
    for name in VARIANTS:
        reduction = values(name, "relative_rmse_reduction")
        aggregate[name] = {
            "mean_relative_rmse_reduction": float(np.mean(reduction)),
            "std_relative_rmse_reduction": float(np.std(reduction, ddof=1)) if len(reduction) > 1 else 0.0,
            "wins": int(sum(bool(row["variants"][name]["lifted_beats_base"]) for row in rows)),
            "n_splits": len(rows),
        }
    primary = aggregate["compact_event_aware"]
    fixed = aggregate["legacy_first_harmonic"]
    selected_rows = [
        row["fit_only_selection"]
        for row in rows
        if row["fit_only_selection"].get("status") == "pass"
        and isinstance(row["fit_only_selection"].get("outer_selected_report"), dict)
    ]
    selected_reduction = np.asarray([
        float(row["outer_selected_report"]["relative_rmse_reduction"])
        for row in selected_rows
    ], dtype=np.float64)
    selected_rmse = np.asarray([
        float(row["outer_selected_report"]["lifted_outcome_rmse"])
        for row in selected_rows
    ], dtype=np.float64)
    selected_base_rmse = np.asarray([
        float(row["outer_selected_report"]["base_outcome_rmse"])
        for row in selected_rows
    ], dtype=np.float64)
    selected_summary = {
        "status": "pass" if len(selected_rows) == len(rows) else "insufficient_support",
        "n_splits": len(selected_rows),
        "mean_relative_rmse_reduction": float(np.mean(selected_reduction)) if len(selected_reduction) else None,
        "std_relative_rmse_reduction": float(np.std(selected_reduction, ddof=1)) if len(selected_reduction) > 1 else 0.0,
        "wins_against_base": int(np.sum(selected_rmse < selected_base_rmse)) if len(selected_rows) else 0,
        "selected_variants": {
            name: sum(row.get("selected_variant") == name for row in selected_rows)
            for name in VARIANTS
        },
    }
    return {
        "artifact": "adaptive_time_frequency_memory_summary",
        "protocol": {
            "replicate_unit": "seeded episode split",
            "row_pooling": False,
            "primary_variant": "compact_event_aware",
            "frozen_negative_control": "legacy_first_harmonic",
            "same_split_and_rank_budget": True,
            "claim_boundary": (
                "A positive result is conditional held-out action-to-task transfer evidence; "
                "it is not causal, contact-mode, planning, or universal-world-model evidence."
            ),
        },
        "n_splits": len(rows),
        "per_split": rows,
        "aggregate": aggregate,
        "fit_only_selected": selected_summary,
        "gates": {
            "primary_event_aware": {
                "status": "pass" if primary["wins"] == len(rows) and primary["mean_relative_rmse_reduction"] > 0 else "fail_or_partial",
                "criterion": "compact raw event lift wins against the current-state/action base on every seeded split",
            },
            "beats_historical_first_harmonic": {
                "status": "pass" if primary["mean_relative_rmse_reduction"] > fixed["mean_relative_rmse_reduction"] else "fail_or_partial",
                "criterion": "compact event reduction exceeds the historical first-harmonic control",
            },
            "fit_only_selector_beats_historical_first_harmonic": {
                "status": (
                    "pass"
                    if selected_summary["status"] == "pass"
                    and selected_summary["mean_relative_rmse_reduction"] > fixed["mean_relative_rmse_reduction"]
                    else "fail_or_partial"
                ),
                "criterion": "fit-only inner episode selector exceeds the historical first-harmonic control",
            },
            "fit_only_selector_beats_base": {
                "status": (
                    "pass"
                    if selected_summary["status"] == "pass"
                    and selected_summary["wins_against_base"] == len(rows)
                    else "fail_or_partial"
                ),
                "criterion": "selected memory family improves held-out transfer on every outer split",
            },
            "standardization_ablation": {
                "status": "pass" if aggregate["compact_event_zscore"]["mean_relative_rmse_reduction"] > 0 else "fail_or_partial",
                "criterion": "fit-only feature standardization must not be treated as an improvement when it harms task transfer",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
