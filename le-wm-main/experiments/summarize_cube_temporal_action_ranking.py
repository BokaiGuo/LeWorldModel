"""Aggregate tie-aware Cube shared-candidate temporal ranking panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def _rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    tolerance = float(payload.get("protocol", {}).get("oracle_tolerance_m", 1.0e-6))
    output = []
    for record in payload.get("held_out", {}).get("records", []):
        actual = np.asarray(record["actual_candidate_error_m"], dtype=np.float64)
        oracle = float(np.min(actual))
        oracle_set = actual <= oracle + tolerance
        row = {"source": str(path), "root_index": int(record["root_index"])}
        methods = [
            method for method, value in record.items()
            if isinstance(value, dict) and "selected_index" in value
        ]
        for method in methods:
            selected = int(record[method]["selected_index"])
            regret = float(actual[selected] - oracle)
            row[method] = {
                "regret_m": regret,
                "oracle_agreement": bool(oracle_set[selected]),
            }
        output.append(row)
    return output


def summarize(paths: list[Path], seed: int) -> dict[str, Any]:
    rows = []
    panels = []
    for path in paths:
        payload = json.loads(path.read_text())
        current = _rows(payload, path)
        rows.extend(current)
        panels.append({"path": str(path), "n_roots": len(current)})
    method_names = sorted({
        method
        for row in rows
        for method in row
        if method not in {"source", "root_index"}
    })
    regret_arrays = {
        method: np.asarray(
            [row[method]["regret_m"] for row in rows], dtype=np.float64
        )
        for method in method_names
    }
    static = regret_arrays.get("latent_static", np.asarray([], dtype=np.float64))
    temporal = regret_arrays.get("latent_temporal", np.asarray([], dtype=np.float64))
    delta = temporal - static if static.size and temporal.size else np.asarray([])
    method_summary = {
        method: {
            "mean_regret_m": float(values.mean()) if values.size else None,
            "bootstrap_ci95": _bootstrap(values, seed + 10 + index),
            "oracle_agreement_rate": float(np.mean([
                row[method]["oracle_agreement"] for row in rows
            ])) if rows else None,
        }
        for index, (method, values) in enumerate(regret_arrays.items())
    }
    paired = {}
    if static.size:
        for index, method in enumerate(method_names):
            if method == "latent_static" or not regret_arrays[method].size:
                continue
            method_delta = regret_arrays[method] - static
            paired[f"{method}_minus_latent_static"] = {
                "mean_regret_delta_m": float(method_delta.mean()),
                "bootstrap_ci95": _bootstrap(method_delta, seed + 100 + index),
                "method_wins": int(np.sum(method_delta < 0.0)),
                "static_wins": int(np.sum(method_delta > 0.0)),
                "ties": int(np.sum(method_delta == 0.0)),
            }
    aggregate = {
        "n_roots": int(len(rows)),
        **method_summary,
        "paired_temporal_minus_static": {
            "mean_regret_delta_m": float(delta.mean()) if delta.size else None,
            "bootstrap_ci95": _bootstrap(delta, seed + 2),
            "temporal_wins": int(np.sum(delta < 0.0)),
            "static_wins": int(np.sum(delta > 0.0)),
            "ties": int(np.sum(delta == 0.0)),
        },
        "paired_against_latent_static": paired,
    }
    return {
        "schema": "cube_temporal_action_ranking_aggregate.v1",
        "panels": panels,
        "aggregate": aggregate,
        "claim_boundary": "Tie-aware guided contact-support ranking diagnostic; no pure RGB online planning claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261046)
    args = parser.parse_args()
    payload = summarize(args.input, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
