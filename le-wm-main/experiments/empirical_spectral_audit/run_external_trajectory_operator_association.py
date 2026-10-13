"""Held-out association between fitted trajectory geometry and h=8 error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _association(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("association requires paired one-dimensional arrays")
    rx, ry = _rank(x), _rank(y)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else 0.0
    return {"n": int(x.size), "pearson": pearson, "spearman": spearman}


def run_report(operator_report: Path, temporal_report: Path, output: Path) -> dict[str, Any]:
    operator = json.loads(operator_report.read_text())
    temporal = json.loads(temporal_report.read_text())
    folds = operator["crossfit"]["folds"]
    records = [
        row
        for row in temporal["records"]
        if row["mode"] == "observed" and int(row["horizon"]) == 8
    ]
    if len(folds) != 21 or len(records) != 42:
        raise ValueError("expected 21 operator folds and 42 observed h=8 temporal records")
    by_episode: dict[int, list[float]] = {}
    for row in records:
        by_episode.setdefault(int(row["episode_id"]), []).append(float(row["mse"]))
    episodes = np.asarray(sorted(by_episode), dtype=np.int64)
    error = np.asarray([np.mean(by_episode[int(episode)]) for episode in episodes], dtype=np.float64)
    fold_by_episode = {int(row["test_episode"]): row for row in folds}
    if set(episodes.tolist()) != set(fold_by_episode):
        raise ValueError("operator and temporal reports have different episode IDs")

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for source in ("true", "predicted"):
        metrics[source] = {
            "one_step_r2": np.asarray([
                fold_by_episode[int(ep)][source]["evaluation"]["one_step_r2"] for ep in episodes
            ]),
            "spectral_radius": np.asarray([
                fold_by_episode[int(ep)][source]["evaluation"]["operator"]["spectral_radius"] for ep in episodes
            ]),
            "singular_to_radius_ratio": np.asarray([
                fold_by_episode[int(ep)][source]["evaluation"]["operator"]["singular_to_radius_ratio"] for ep in episodes
            ]),
            "h8_gain_over_radius_power": np.asarray([
                fold_by_episode[int(ep)][source]["evaluation"]["operator"]["operator_powers"]["8"]["gain_over_radius_power"] for ep in episodes
            ]),
            "h8_composition_defect": np.asarray([
                fold_by_episode[int(ep)][source]["evaluation"]["composition_defect"]["8"] for ep in episodes
            ]),
        }
    associations = {}
    for source, source_metrics in metrics.items():
        associations[source] = {
            metric: _association(values, error) for metric, values in source_metrics.items()
        }
    report = {
        "artifact": "external_frozen_lewm_trajectory_operator_heldout_association",
        "status": "complete",
        "protocol": {
            "operator_report": str(operator_report),
            "temporal_report": str(temporal_report),
            "unit": "one held-out episode per leave-one-episode-out fold; outcome is mean observed-action h=8 latent MSE over that episode's two windows",
            "n_episodes": int(episodes.size),
            "normalization": "public",
            "association": "descriptive Pearson and Spearman correlations; no p-value or causal interpretation",
        },
        "outcome": {
            "episodes": episodes.tolist(),
            "h8_mse_mean": float(error.mean()),
            "h8_mse_std": float(error.std(ddof=1)),
        },
        "associations": associations,
        "claim_boundary": [
            "Each geometry statistic is fitted without the held-out episode, so this is an out-of-episode association check rather than an in-sample correlation.",
            "The sample contains 21 episodes and the correlations are descriptive; they do not establish causal error amplification, control, planning, or universal operator behavior.",
        ],
    }
    if not _finite(report):
        raise ValueError("association report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Held-out association: trajectory operator geometry versus external h=8 error",
        "",
        "Each geometry statistic is fitted on the other episodes; the outcome is the mean observed-action h=8 latent MSE on the held-out episode. Correlations are descriptive.",
        "",
        "| trajectory source | geometry statistic | Pearson | Spearman | n episodes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for source, values in report["associations"].items():
        for metric, association in values.items():
            lines.append(
                f"| {source} | {metric} | {association['pearson']:.6g} | {association['spearman']:.6g} | {association['n']} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-report", type=Path, required=True)
    parser.add_argument("--temporal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.operator_report, args.temporal_report, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
