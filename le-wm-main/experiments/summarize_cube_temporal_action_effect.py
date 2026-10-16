"""Aggregate independent Cube temporal action-effect panels by held-out root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "rgb_static",
    "rgb_temporal",
    "rgb_temporal_interaction",
    "latent_static",
    "latent_temporal",
    "latent_temporal_interaction",
    "rgb_static_plus_event_support_interaction_gate",
)

BASELINE_FOR = {
    "rgb_static": "rgb_static",
    "rgb_temporal": "rgb_static",
    "rgb_temporal_interaction": "rgb_static",
    "latent_static": "latent_static",
    "latent_temporal": "latent_static",
    "latent_temporal_interaction": "latent_static",
    "rgb_static_plus_event_support_interaction_gate": "rgb_static",
}


def _bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def summarize(paths: list[Path], seed: int) -> dict[str, Any]:
    rows: dict[str, list[float]] = {method: [] for method in METHODS}
    panel_summaries = []
    for panel_id, path in enumerate(paths):
        payload = json.loads(path.read_text())
        panel = {"path": str(path), "schema": payload.get("schema"), "methods": {}}
        for method in METHODS:
            root_rows = payload.get("metrics", {}).get(method, {}).get("test_root_rmse", [])
            values = np.asarray([row["rmse"] for row in root_rows], dtype=np.float64)
            rows[method].extend(values.tolist())
            panel["methods"][method] = {
                "n_roots": int(values.size),
                "mean_rmse": float(values.mean()) if values.size else None,
            }
        panel_summaries.append(panel)
    aggregate: dict[str, Any] = {}
    for offset, method in enumerate(METHODS):
        values = np.asarray(rows[method], dtype=np.float64)
        baseline_name = BASELINE_FOR[method]
        baseline = np.asarray(rows[baseline_name], dtype=np.float64)
        delta = values - baseline if method != baseline_name and values.size > 0 and values.size == baseline.size else None
        aggregate[method] = {
            "n_test_roots": int(values.size),
            "mean_rmse": float(values.mean()) if values.size else None,
            "bootstrap_ci95": _bootstrap(values, seed + offset) if values.size else None,
            "baseline_method": baseline_name,
            "delta_vs_baseline_mean_rmse": float(delta.mean()) if delta is not None else 0.0,
            "delta_vs_baseline_bootstrap_ci95": _bootstrap(delta, seed + 100 + offset) if delta is not None else [0.0, 0.0],
            "wins_vs_baseline": int(np.sum(delta < 0.0)) if delta is not None else 0,
            "losses_vs_baseline": int(np.sum(delta > 0.0)) if delta is not None else 0,
        }
    return {
        "schema": "cube_temporal_action_effect_aggregate.v1",
        "panels": panel_summaries,
        "aggregate": aggregate,
        "claim_boundary": "Root-aggregated effect-prediction diagnostic; no online planning or causal-dynamics claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    payload = summarize(args.input, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
