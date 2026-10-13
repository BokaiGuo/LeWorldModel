"""Summarize horizon-conditioned memory diagnostics without re-selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_lifted_signal_information_audit import (
    adaptive_time_frequency_features,
)
from experiments.empirical_spectral_audit.run_real_enriched_hybrid_audit import (
    fit_only_pod_projection,
    load_enriched_arrays,
)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _paired_bootstrap_mean_ci(values: list[float], *, seed: int = 20261013, draws: int = 20000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "lower_95": None, "upper_95": None}
    values_array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values_array), size=(int(draws), len(values_array)))
    means = values_array[indices].mean(axis=1)
    return {
        "mean": float(values_array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _support_descriptors(
    arrays: dict[str, Any], report: dict[str, Any], history_length: int, state_rank: int
) -> dict[str, float | None]:
    fit_index = np.asarray(report["split"]["fit_episode_ids"])
    episode_all = np.asarray(arrays["episode_id"])
    fit_index = np.flatnonzero(np.isin(episode_all, fit_index))
    projected, _projection, _basis, _center = fit_only_pod_projection(
        np.asarray(arrays["source_latent"], dtype=np.float64), fit_index, rank=state_rank
    )
    features = adaptive_time_frequency_features(
        projected,
        np.asarray(arrays["action"], dtype=np.float64),
        np.asarray(arrays["episode_id"]),
        np.asarray(arrays["source_frame"]),
        history_length=history_length,
    )
    indices = np.asarray(features["indices"], dtype=np.int64)
    episode_id = np.asarray(arrays["episode_id"])[indices]
    fit_eps = set(int(x) for x in report["split"]["fit_episode_ids"])
    out_eps = set(int(x) for x in report["split"]["outcome_episode_ids"])
    fit = np.isin(episode_id, list(fit_eps))
    outcome = np.isin(episode_id, list(out_eps))
    score = np.asarray(features["event_score"], dtype=np.float64)
    action = np.asarray(arrays["action"], dtype=np.float64)[indices]
    fit_threshold = float(np.median(score[fit])) if np.any(fit) else float("nan")
    return {
        "fit_event_score_mean": _mean(score[fit].tolist()),
        "outcome_event_score_mean": _mean(score[outcome].tolist()),
        "fit_event_density_above_fit_median": (
            float(np.mean(score[fit] >= fit_threshold)) if np.any(fit) else None
        ),
        "outcome_event_density_above_fit_median": (
            float(np.mean(score[outcome] >= fit_threshold)) if np.any(outcome) else None
        ),
        "fit_action_norm_mean": _mean(np.linalg.norm(action[fit], axis=1).tolist()),
        "outcome_action_norm_mean": _mean(np.linalg.norm(action[outcome], axis=1).tolist()),
        "fit_action_cov_trace": (
            float(np.trace(np.cov(action[fit].T))) if int(np.sum(fit)) > 1 else None
        ),
    }


def summarize(paths: list[Path], source: Path, history_length: int = 4) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one report is required")
    arrays_by_horizon: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if report.get("status") != "available":
            rows.append({"path": str(path), "status": report.get("status")})
            continue
        horizon = int(report["protocol"]["horizon"])
        arrays_by_horizon.setdefault(horizon, load_enriched_arrays(source, horizon=horizon))
        selector = report.get("selector_diagnostic", {})
        instability = report.get("selection_instability", {})
        selector_report = selector.get("selected_outcome_report")
        frozen_report = report["frozen_horizon_policy"]["outcome_report"]
        carrier_report = report["candidate_reports"]["fixed_fourier"]
        selected_variant = selector.get("selected_variant")
        selected_rmse = None if selector_report is None else float(selector_report["lifted_outcome_rmse"])
        frozen_rmse = float(frozen_report["lifted_outcome_rmse"])
        split = report["split"]
        row = {
            "path": str(path),
            "horizon": horizon,
            "split_seed": split.get("split_seed"),
            "frozen_variant": report["frozen_horizon_policy"]["variant"],
            "selector_variant": selected_variant,
            "selector_stability": instability.get("stability"),
            "selector_normalized_entropy": instability.get("normalized_entropy"),
            "selection_frequency": instability.get("selection_frequency", {}),
            "fallback": not bool(selector.get("stable", False)),
            "selector_outcome_rmse": selected_rmse,
            "frozen_outcome_rmse": frozen_rmse,
            "carrier_outcome_rmse": float(carrier_report["lifted_outcome_rmse"]),
            "frozen_minus_carrier_regret": float(
                frozen_rmse - float(carrier_report["lifted_outcome_rmse"])
            ),
            "selector_minus_frozen_regret": (
                None if selected_rmse is None else float(selected_rmse - frozen_rmse)
            ),
        }
        row.update(_support_descriptors(
            arrays_by_horizon[horizon], report, history_length,
            int(report["protocol"].get("state_rank", 32)),
        ))
        rows.append(row)

    by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "horizon" in row:
            by_horizon[int(row["horizon"])].append(row)
    aggregate: dict[str, Any] = {}
    for horizon, horizon_rows in sorted(by_horizon.items()):
        selected = [str(r["selector_variant"]) for r in horizon_rows if r.get("selector_variant")]
        regrets = [float(r["selector_minus_frozen_regret"]) for r in horizon_rows if r.get("selector_minus_frozen_regret") is not None]
        carrier_regrets = [float(r["frozen_minus_carrier_regret"]) for r in horizon_rows]
        stability = [float(r["selector_stability"]) for r in horizon_rows if r.get("selector_stability") is not None]
        frozen_variants = Counter(str(r["frozen_variant"]) for r in horizon_rows)
        aggregate[str(horizon)] = {
            "split_count": len(horizon_rows),
            "frozen_policy": dict(frozen_variants),
            "selector_variant_frequency": dict(Counter(selected)),
            "fallback_count": int(sum(bool(r["fallback"]) for r in horizon_rows)),
            "fallback_frequency": float(np.mean([bool(r["fallback"]) for r in horizon_rows])),
            "selector_stability_mean": _mean(stability),
            "selector_stability_min": min(stability) if stability else None,
            "selector_minus_frozen_regret_mean": _mean(regrets),
            "selector_minus_frozen_regret_positive_count": int(sum(value > 0 for value in regrets)),
            "frozen_minus_carrier_regret_mean": _mean(carrier_regrets),
            "frozen_minus_carrier_regret_paired_bootstrap_ci": _paired_bootstrap_mean_ci(
                carrier_regrets, seed=20261013 + horizon
            ),
            "frozen_beats_carrier_count": int(sum(value < 0 for value in carrier_regrets)),
            "frozen_beats_carrier_frequency": float(np.mean([value < 0 for value in carrier_regrets])),
            "fit_event_score_mean": _mean([float(r["fit_event_score_mean"]) for r in horizon_rows if r["fit_event_score_mean"] is not None]),
            "outcome_event_score_mean": _mean([float(r["outcome_event_score_mean"]) for r in horizon_rows if r["outcome_event_score_mean"] is not None]),
        }
    return {
        "artifact": "horizon_conditioned_transfer_sufficiency_summary",
        "status": "available",
        "protocol": {
            "history_length": int(history_length),
            "selector_is_diagnostic_only": True,
            "deployment_policy": "predeclared_horizon_policy",
            "claim_boundary": "transfer-sufficiency diagnostic; no action-ranking or planning claim",
        },
        "aggregate_by_horizon": aggregate,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-length", type=int, default=4)
    args = parser.parse_args()
    result = summarize(args.input, args.source, history_length=args.history_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate_by_horizon"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
