"""Summarize preregistered recurrent-memory split reports.

The summary never selects a split, checkpoint, or model by held-out planning
performance.  It reports all supplied splits, paired RMSE deltas, and the
predeclared Fourier-comparison gate.  A positive local prediction result is
not promoted to action-ranking or online-planning evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sd": float("nan")}
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one split report is required")
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("status") != "available":
            rows.append({"path": str(path), "status": report.get("status")})
            continue
        legacy = float(report["controls"]["legacy_first_harmonic"]["lifted_outcome_rmse"])
        fixed = float(report["controls"]["fixed_fourier"]["lifted_outcome_rmse"])
        gru = float(report["recurrent"]["gru"]["outcome_rmse"])
        gated = float(report["recurrent"]["event_gated_gru"]["outcome_rmse"])
        rows.append({
            "path": str(path),
            "split_seed": report["split"].get("split_seed"),
            "horizon": report["protocol"].get("horizon"),
            "state_rank": report["protocol"].get("state_rank"),
            "fit_episodes": report["support"]["fit_episodes"],
            "outcome_episodes": report["support"]["outcome_episodes"],
            "fit_sequences": report["support"]["fit_sequences"],
            "outcome_sequences": report["support"]["outcome_sequences"],
            "legacy_fourier_rmse": legacy,
            "fixed_fourier_rmse": fixed,
            "gru_rmse": gru,
            "event_gated_gru_rmse": gated,
            "gru_minus_legacy_fourier": gru - legacy,
            "event_gated_minus_legacy_fourier": gated - legacy,
            "event_gated_minus_gru": gated - gru,
            "gru_beats_legacy_fourier": bool(gru < legacy),
            "event_gated_beats_legacy_fourier": bool(gated < legacy),
            "event_gated_beats_gru": bool(gated < gru),
            "selector_variant": report.get("fit_only_spectral_selector", {}).get("selected_variant"),
            "selector_rmse": (
                None if report.get("fit_only_spectral_selector", {}).get("outer_selected_report") is None
                else float(report["fit_only_spectral_selector"]["outer_selected_report"]["lifted_outcome_rmse"])
            ),
            "frozen_policy_variant": report.get("frozen_horizon_policy", {}).get("variant"),
            "frozen_policy_rmse": (
                None if report.get("frozen_horizon_policy", {}).get("outcome_report") is None
                else float(report["frozen_horizon_policy"]["outcome_report"]["lifted_outcome_rmse"])
            ),
        })
    available = [row for row in rows if row.get("status", "available") == "available"]
    if not available:
        return {
            "artifact": "event_gated_recurrent_memory_summary",
            "status": "unavailable",
            "rows": rows,
        }
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in available]

    selector_rows = [row for row in available if row.get("selector_rmse") is not None]
    selector_values = [float(row["selector_rmse"]) for row in selector_rows]
    selector_vs_legacy = [float(row["selector_rmse"]) - float(row["legacy_fourier_rmse"]) for row in selector_rows]
    selector_vs_fixed = [float(row["selector_rmse"]) - float(row["fixed_fourier_rmse"]) for row in selector_rows]
    frozen_rows = [row for row in available if row.get("frozen_policy_rmse") is not None]
    frozen_values = [float(row["frozen_policy_rmse"]) for row in frozen_rows]
    frozen_vs_fixed = [float(row["frozen_policy_rmse"]) - float(row["fixed_fourier_rmse"]) for row in frozen_rows]
    frozen_vs_legacy = [float(row["frozen_policy_rmse"]) - float(row["legacy_fourier_rmse"]) for row in frozen_rows]

    state_ranks = sorted({row.get("state_rank") for row in available})
    return {
        "artifact": "event_gated_recurrent_memory_summary",
        "status": "available",
        "protocol": {
            "split_count": len(available),
            "episode_disjoint": True,
            "fit_only_projection_and_normalization": True,
            "state_ranks": state_ranks,
            "claim_boundary": (
                "held-out action-to-task prediction diagnostic only; no action-ranking, "
                "contact-closure, online-planning, or causal-identification claim"
            ),
        },
        "rows": rows,
        "aggregate": {
            "legacy_fourier_rmse": _mean_std(values("legacy_fourier_rmse")),
            "fixed_fourier_rmse": _mean_std(values("fixed_fourier_rmse")),
            "gru_rmse": _mean_std(values("gru_rmse")),
            "event_gated_gru_rmse": _mean_std(values("event_gated_gru_rmse")),
            "gru_minus_legacy_fourier": _mean_std(values("gru_minus_legacy_fourier")),
            "event_gated_minus_legacy_fourier": _mean_std(values("event_gated_minus_legacy_fourier")),
            "event_gated_minus_gru": _mean_std(values("event_gated_minus_gru")),
            "selector_rmse": _mean_std(selector_values),
            "selector_minus_legacy_fourier": _mean_std(selector_vs_legacy),
            "selector_minus_fixed_fourier": _mean_std(selector_vs_fixed),
            "frozen_policy_rmse": _mean_std(frozen_values),
            "frozen_policy_minus_fixed_fourier": _mean_std(frozen_vs_fixed),
            "frozen_policy_minus_legacy_fourier": _mean_std(frozen_vs_legacy),
        },
        "gates": {
            "gru_beats_legacy_fourier_all_splits": all(row["gru_beats_legacy_fourier"] for row in available),
            "event_gated_beats_legacy_fourier_all_splits": all(row["event_gated_beats_legacy_fourier"] for row in available),
            "event_gated_beats_gru_all_splits": all(row["event_gated_beats_gru"] for row in available),
            "gru_wins_vs_legacy_fourier": sum(row["gru_beats_legacy_fourier"] for row in available),
            "event_gated_wins_vs_legacy_fourier": sum(row["event_gated_beats_legacy_fourier"] for row in available),
            "event_gated_wins_vs_gru": sum(row["event_gated_beats_gru"] for row in available),
            "selector_available": len(selector_rows) == len(available),
            "selector_beats_legacy_fourier_all_splits": bool(selector_rows) and all(
                row["selector_rmse"] < row["legacy_fourier_rmse"] for row in selector_rows
            ),
            "selector_wins_vs_legacy_fourier": sum(
                row["selector_rmse"] < row["legacy_fourier_rmse"] for row in selector_rows
            ),
            "selector_beats_fixed_fourier_all_splits": bool(selector_rows) and all(
                row["selector_rmse"] < row["fixed_fourier_rmse"] for row in selector_rows
            ),
            "selector_wins_vs_fixed_fourier": sum(
                row["selector_rmse"] < row["fixed_fourier_rmse"] for row in selector_rows
            ),
            "selector_variants": {
                str(name): sum(row.get("selector_variant") == name for row in selector_rows)
                for name in sorted({row.get("selector_variant") for row in selector_rows})
            },
            "frozen_policy_available": len(frozen_rows) == len(available),
            "frozen_policy_beats_fixed_fourier_all_splits": bool(frozen_rows) and all(
                row["frozen_policy_rmse"] < row["fixed_fourier_rmse"] for row in frozen_rows
            ),
            "frozen_policy_wins_vs_fixed_fourier": sum(
                row["frozen_policy_rmse"] < row["fixed_fourier_rmse"] for row in frozen_rows
            ),
            "frozen_policy_beats_legacy_fourier_all_splits": bool(frozen_rows) and all(
                row["frozen_policy_rmse"] < row["legacy_fourier_rmse"] for row in frozen_rows
            ),
            "frozen_policy_variants": {
                str(name): sum(row.get("frozen_policy_variant") == name for row in frozen_rows)
                for name in sorted({row.get("frozen_policy_variant") for row in frozen_rows})
            },
            "task_utilization": "not_established; no independent action-ranking or planning outcome supplied",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "gates": report.get("gates", {})}, indent=2))


if __name__ == "__main__":
    main()
