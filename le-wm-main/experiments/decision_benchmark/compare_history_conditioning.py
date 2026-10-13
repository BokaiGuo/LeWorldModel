"""Paired comparison of single-frame and causal-history RGB successor reports.

The comparison is root-paired and fail-closed: the two reports must contain
the same root/candidate keys, and uncertainty is computed over roots rather
than treating the 16 candidates from one root as independent experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_values(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    values = {}
    for root, rows in grouped.items():
        predicted = np.asarray([row["predicted_vs_realized_projection_rmse"] for row in rows], dtype=np.float64)
        baseline = np.asarray([
            np.sqrt(np.mean((np.asarray(row["source_projection"], dtype=np.float64)
                             - np.asarray(row["realized_terminal_projection"], dtype=np.float64)) ** 2))
            for row in rows
        ])
        values[root] = {
            "predicted_rmse": float(np.mean(predicted)),
            "source_baseline_rmse": float(np.mean(baseline)),
            "relative_improvement": float(1.0 - np.mean(predicted) / max(np.mean(baseline), 1e-12)),
            "candidate_count": int(len(rows)),
        }
    return values


def _bootstrap(values: np.ndarray, seed: int, samples: int = 5000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(values[rng.integers(0, len(values), len(values))])
        for _ in range(samples)
    ])
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "roots": int(len(values)),
    }


def _sign_flip(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 20:
        rng = np.random.default_rng(20261014)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, len(values)))
        draws = np.mean(signs * values.reshape(1, -1), axis=1)
    else:
        masks = np.arange(1 << len(values), dtype=np.uint64)[:, None]
        signs = np.where((masks >> np.arange(len(values), dtype=np.uint64)) & 1, 1.0, -1.0)
        draws = np.mean(signs * values.reshape(1, -1), axis=1)
    observed = float(np.mean(values))
    return {
        "observed_mean": observed,
        "p_two_sided": float(np.mean(np.abs(draws) >= abs(observed))),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def compare(single_report: Path, history_report: Path, output: Path) -> dict[str, Any]:
    single = _root_values(_load(single_report))
    history = _root_values(_load(history_report))
    if set(single) != set(history):
        raise ValueError(f"root mismatch: single={sorted(single)}, history={sorted(history)}")
    roots = sorted(single)
    for root in roots:
        if single[root]["candidate_count"] != history[root]["candidate_count"]:
            raise ValueError(f"candidate count mismatch for {root}")
    delta_rmse = np.asarray([history[root]["predicted_rmse"] - single[root]["predicted_rmse"] for root in roots])
    delta_improvement = np.asarray([history[root]["relative_improvement"] - single[root]["relative_improvement"] for root in roots])
    rows = [{
        "instance_id": root,
        "single_frame": single[root],
        "history_conditioned": history[root],
        "history_minus_single_predicted_rmse": float(delta_rmse[index]),
        "history_minus_single_relative_improvement": float(delta_improvement[index]),
    } for index, root in enumerate(roots)]
    report = {
        "schema": "paired_history_conditioning_rgb_successor.v1",
        "protocol": {
            "pairing_unit": "root",
            "candidate_bank": "same candidate family, same source window, same outcome roots, same 16 candidates per root",
            "single_frame_report": str(single_report.resolve()),
            "history_report": str(history_report.resolve()),
            "uncertainty": "root bootstrap and exact sign-flip",
        },
        "rows": rows,
        "summary": {
            "root_count": len(roots),
            "single_frame": {
                "predicted_rmse": _bootstrap(np.asarray([single[root]["predicted_rmse"] for root in roots]), 20261015),
                "relative_improvement": _bootstrap(np.asarray([single[root]["relative_improvement"] for root in roots]), 20261016),
            },
            "history_conditioned": {
                "predicted_rmse": _bootstrap(np.asarray([history[root]["predicted_rmse"] for root in roots]), 20261017),
                "relative_improvement": _bootstrap(np.asarray([history[root]["relative_improvement"] for root in roots]), 20261018),
            },
            "history_minus_single": {
                "predicted_rmse": {"bootstrap": _bootstrap(delta_rmse, 20261019), "sign_flip": _sign_flip(delta_rmse)},
                "relative_improvement": {"bootstrap": _bootstrap(delta_improvement, 20261020), "sign_flip": _sign_flip(delta_improvement)},
            },
        },
        "interpretation": "Negative history-minus-single RMSE is favorable; positive history-minus-single relative improvement is favorable.",
        "claim_boundary": "Paired realized-RGB successor evidence on PointMaze outcome roots; not closed-loop online planning or universal transfer.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-report", type=Path, required=True)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.single_report, args.history_report, args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
