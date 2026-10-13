"""Data-efficiency sweep for equivariance/alignment interventions."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.regularizers import operator_alignment_diagnostic
    from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import (
        EquivarianceBenchmarkConfig,
        _composition_error,
        _equivariance_metrics,
        _fit_joint,
        _rollout_mse,
    )
except ModuleNotFoundError:  # direct script invocation
    from regularizers import operator_alignment_diagnostic
    from run_alignment_equivariance_joint_synthetic import (
        EquivarianceBenchmarkConfig,
        _composition_error,
        _equivariance_metrics,
        _fit_joint,
        _rollout_mse,
    )


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1 if array.size > 1 else 0))}


def run_one(
    sample_count: int,
    mode: str,
    predictor_weight: float,
    alignment_weight: float,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    config = EquivarianceBenchmarkConfig(
        train_samples=int(sample_count),
        validation_samples=max(128, int(sample_count // 2)),
        test_samples=max(256, int(sample_count)),
        epochs=epochs,
    )
    model, _, _, test, rollout, _, best_epoch, best_val = _fit_joint(
        mode, predictor_weight, alignment_weight, seed, config
    )
    with torch.no_grad():
        source = model.encode(test["state"])
        target = model.encode(test["target"])
        prediction = model.predict(source, test["action"])
        equivariance = _equivariance_metrics(model, test)
        alignment = operator_alignment_diagnostic(
            source, prediction, actions=test["action"], mode="conditional", num_bins=2, min_samples=32, ridge=1e-3
        )
    return {
        "sample_count": int(sample_count),
        "mode": mode,
        "predictor_weight": float(predictor_weight),
        "alignment_weight": float(alignment_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val),
        "test_one_step_mse": float((prediction - target).square().mean()),
        "rollout_h8_mse": float(_rollout_mse(model, rollout, (8,))["8"]),
        "composition_error": float(_composition_error(model, rollout["states"][:, 0], config)),
        "predictor_equivariance_error": float(equivariance["predictor_equivariance_error"]),
        "operator_alignment": float(alignment["operator_alignment"]),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = f"n={row['sample_count']}:mode={row['mode']}:predictor={row['predictor_weight']}:alignment={row['alignment_weight']}"
        groups.setdefault(key, []).append(row)
    metrics = (
        "test_one_step_mse",
        "rollout_h8_mse",
        "composition_error",
        "predictor_equivariance_error",
        "operator_alignment",
    )
    return {
        key: {
            "n_seeds": len(rows),
            "sample_count": rows[0]["sample_count"],
            "mode": rows[0]["mode"],
            "predictor_weight": rows[0]["predictor_weight"],
            "alignment_weight": rows[0]["alignment_weight"],
            **{metric: _mean_std([float(row[metric]) for row in rows]) for metric in metrics},
        }
        for key, rows in groups.items()
    }


def run_report(sample_counts: list[int], seeds: list[int], epochs: int = 120) -> dict[str, Any]:
    conditions = [("none", 0.0, 0.0), ("predictor", 1.0, 0.0), ("both", 1.0, 0.003)]
    records = []
    started = time.time()
    for sample_count in sample_counts:
        for mode, predictor_weight, alignment_weight in conditions:
            for seed in seeds:
                print(f"[data-efficiency] n={sample_count} mode={mode} alignment={alignment_weight} seed={seed}")
                records.append(run_one(sample_count, mode, predictor_weight, alignment_weight, seed, epochs))
    report = {
        "protocol": {
            "sample_counts": [int(value) for value in sample_counts],
            "seeds": [int(seed) for seed in seeds],
            "conditions": conditions,
            "checkpoint_selection": "held-out one-step validation MSE",
            "same_protocol_across_sample_counts": True,
            "synthetic_vector_world_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": _summary(records),
        "claim_boundary": [
            "data-efficiency results are synthetic mechanism evidence",
            "no LeWM, PushT, online planning, or causal-control evidence",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Data-efficiency sweep for structural interventions",
        "",
        "All conditions use matched seeds and held-out one-step checkpoint selection. Values are mean ± standard deviation.",
        "",
        "| train samples | condition | seeds | one-step MSE | rollout h=8 | composition | predictor equivariance | operator alignment |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for values in report["summary"].values():
        fmt = lambda metric: f"{values[metric]['mean']:.6g} ± {values[metric]['std']:.2g}"
        condition = f"{values['mode']} ({values['predictor_weight']},{values['alignment_weight']})"
        lines.append(
            f"| {values['sample_count']} | {condition} | {values['n_seeds']} | {fmt('test_one_step_mse')} | "
            f"{fmt('rollout_h8_mse')} | {fmt('composition_error')} | {fmt('predictor_equivariance_error')} | {fmt('operator_alignment')} |"
        )
    lines.extend(["", "## Boundary", "", "This is synthetic mechanism evidence only."])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    report = run_report(args.sample_counts, args.seeds, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
