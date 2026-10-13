"""Strict factor-matched audit for conditional transfer-risk selection.

This audit consumes immutable ``real_checkpoint_selection.v1`` records.  It
does not tune on held-out planning loss.  Candidates are first stratified by
architecture and training stage, then filtered using diagnostic-split
prediction MSE and a full local-Jacobian spectrum fingerprint.  Only after
those filters are applied are task-metric alignment and task-space geometry
used as selectors.

The result is deliberately fail-closed: a stratum with fewer than two
quality-and-spectrum-matched candidates is reported as non-identifiable.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


TASK_FIELDS = (
    "task_metric_alignment",
    "task_metric_commutator",
    "task_metric_departure_from_normality",
    "task_metric_transient_amplification",
    "task_metric_spectral_radius",
)


def _stage(record: dict[str, Any]) -> int:
    factors = record.get("model_factors", {})
    if "training_stage" in factors:
        return int(factors["training_stage"])
    match = re.search(r"epoch[_/](\d+)", str(record.get("checkpoint", "")))
    if match is None:
        raise ValueError(f"cannot infer training stage from {record.get('checkpoint')!r}")
    return int(match.group(1))


def _spectrum(record: dict[str, Any]) -> np.ndarray:
    values = record.get("diagnostics", {}).get("predictor_spectrum")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{record.get('checkpoint')!r} lacks full predictor_spectrum")
    array = np.asarray([complex(float(item["real"]), float(item["imag"])) for item in values])
    return array[np.lexsort((array.imag, array.real))]


def _spectrum_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-12)
    return float(np.max(np.abs(left - right)) / scale)


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_strata": 0}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, array.size, size=(4000, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n_strata": int(array.size),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    result[order] = np.arange(values.size, dtype=np.float64)
    return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError(f"{path} has incompatible record_schema")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"{path} has no record")
    missing = [name for name in TASK_FIELDS if name not in record.get("diagnostics", {})]
    if missing:
        raise ValueError(f"{path} lacks task diagnostics: {missing}")
    # These are the declared common factors.  Runtime action distributions and
    # coordinate gauges are checked across the whole input below.
    _ = _stage(record)
    _ = _spectrum(record)
    # Selection records keep the immutable split/interface declaration at the
    # payload level.  Carry it into this audit object so protocol invariants
    # are checked rather than silently assumed.
    record["protocol"] = payload.get("protocol", {})
    return record


def audit(
    records: list[dict[str, Any]],
    *,
    mse_tolerance: float = 0.05,
    spectrum_tolerance: float = 0.05,
    bootstrap_seed: int = 20261011,
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("at least two records are required")
    protocols = [item["protocol"] for item in records]
    gauges = {item.get("coordinate_gauge") for item in protocols}
    actions = {
        item.get("prediction_quality", {}).get("action_distribution")
        for item in protocols
    }
    environments = {item.get("environment") for item in protocols}
    if len(gauges) != 1 or len(actions) != 1:
        raise ValueError(f"coordinate/action protocol is not fixed: gauges={gauges}, actions={actions}")
    factor_dims = {int(item.get("model_factors", {}).get("latent_dimension", -1)) for item in records}
    if len(factor_dims) != 1 or -1 in factor_dims:
        raise ValueError(f"latent dimension is not fixed: {factor_dims}")

    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        group = str(item["group"])
        variant = group.split("/")[-2] if "/" in group else group
        strata[(variant, _stage(item))].append(item)

    reports: list[dict[str, Any]] = []
    regrets: dict[str, list[float]] = defaultdict(list)
    unavailable: dict[str, int] = defaultdict(int)
    for (variant, stage), candidates in sorted(strata.items()):
        mse = np.asarray([float(item["diagnostics"]["prediction_mse"]) for item in candidates])
        losses = np.asarray([float(item["heldout"]["planning_loss"]) for item in candidates])
        anchor = _spectrum(candidates[int(np.argmin(mse))])
        distances = np.asarray([_spectrum_distance(anchor, _spectrum(item)) for item in candidates])
        mse_floor = float(mse.min())
        eligible = np.flatnonzero(
            (mse <= mse_floor * (1.0 + mse_tolerance)) & (distances <= spectrum_tolerance)
        )
        selectors: dict[str, int | None] = {
            "task_metric_alignment": None,
            "task_metric_geometry": None,
            "prediction_mse": None,
            "random_expected": None,
        }
        if eligible.size >= 2:
            def values(name: str) -> np.ndarray:
                return np.asarray([float(candidates[index]["diagnostics"][name]) for index in eligible])

            selectors["task_metric_alignment"] = int(eligible[np.argmin(values("task_metric_alignment"))])
            geometry = np.mean(np.stack([_rank(values(name)) for name in TASK_FIELDS]), axis=0)
            selectors["task_metric_geometry"] = int(eligible[np.argmin(geometry)])
            selectors["prediction_mse"] = int(eligible[np.argmin(mse[eligible])])
            selectors["random_expected"] = -1
            oracle = int(eligible[np.argmin(losses[eligible])])
            oracle_loss = float(losses[oracle])
            for name, index in selectors.items():
                if name == "random_expected":
                    value = float(losses[eligible].mean() - oracle_loss)
                else:
                    value = float(losses[int(index)] - oracle_loss)
                regrets[name].append(value)
        else:
            oracle = None
            oracle_loss = None
            for name in selectors:
                unavailable[name] += 1
        reports.append({
            "variant": variant,
            "training_stage": stage,
            "environment": sorted(environments),
            "candidate_count": len(candidates),
            "eligible_count": int(eligible.size),
            "eligible_checkpoints": [candidates[int(index)]["checkpoint"] for index in eligible],
            "max_spectrum_distance_to_mse_anchor": float(distances.max()),
            "oracle_checkpoint": None if oracle is None else candidates[oracle]["checkpoint"],
            "oracle_loss": oracle_loss,
            "selector_indices": selectors,
        })
    return {
        "protocol": {
            "estimand": "conditional transfer-risk selection after quality, latent dimension, stage, action distribution, and local-Jacobian spectrum matching",
            "quality_filter": f"prediction_mse <= min * (1 + {mse_tolerance}) on diagnostic split",
            "spectrum_filter": f"max matched eigenvalue distance / max eigenvalue <= {spectrum_tolerance}",
            "selectors": ["task_metric_alignment", "task_metric_geometry", "prediction_mse", "random_expected"],
            "outcome": "held-out planning loss only",
            "environments": sorted(environments),
            "latent_dimensions": sorted(factor_dims),
            "coordinate_gauges": sorted(gauges),
            "action_distributions": sorted(actions),
        },
        "strata": reports,
        "summary": {
            "selection_regret": {
                name: _bootstrap(values, bootstrap_seed + index)
                for index, (name, values) in enumerate(sorted(regrets.items()))
            },
            "unavailable_strata": dict(sorted(unavailable.items())),
            "identifiable_strata": sum(1 for row in reports if row["eligible_count"] >= 2),
            "total_strata": len(reports),
        },
        "claim_boundary": (
            "This audit can support only a conditional transfer-risk result when the eligible strata are replicated across independent architectures, seeds, and environments. "
            "It cannot establish a universal certificate from a single environment or from strata with no fixed-spectrum matched pair."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mse-tolerance", type=float, default=0.05)
    parser.add_argument("--spectrum-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    report = audit(
        [_load(path) for path in args.record],
        mse_tolerance=args.mse_tolerance,
        spectrum_tolerance=args.spectrum_tolerance,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "conditional_transfer_selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
