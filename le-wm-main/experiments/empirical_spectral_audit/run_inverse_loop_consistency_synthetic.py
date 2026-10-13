"""Inverse-action and closed-loop consistency audit in the additive vector world."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import (
        EquivarianceBenchmarkConfig,
        _fit_joint,
        _rollout_mse,
    )
except ModuleNotFoundError:  # direct script invocation
    from run_alignment_equivariance_joint_synthetic import EquivarianceBenchmarkConfig, _fit_joint, _rollout_mse


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _normalized_rmse(actual: torch.Tensor, target: torch.Tensor) -> float:
    denominator = target.square().mean().sqrt().clamp_min(1e-8)
    return float((actual - target).square().mean().sqrt() / denominator)


@torch.no_grad()
def _loop_metrics(model, seed: int, samples: int = 2048) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 910_001)
    base = torch.randn(samples, 4, generator=generator)
    first = 0.25 * torch.randn(samples, 4, generator=generator)
    second = 0.25 * torch.randn(samples, 4, generator=generator)
    initial = model.encode(base)

    forward = model.predict(initial, first)
    inverse = model.predict(forward, -first)
    inverse_error = _normalized_rmse(inverse, initial)

    current = initial
    for action in (first, second, -first, -second):
        current = model.predict(current, action)
    loop_error = _normalized_rmse(current, initial)

    third = 0.25 * torch.randn(samples, 4, generator=generator)
    current = initial
    for action in (first, second, third, -first, -second, -third):
        current = model.predict(current, action)
    loop6_error = _normalized_rmse(current, initial)
    return {
        "inverse_pair_error": inverse_error,
        "closed_loop_h4_error": loop_error,
        "closed_loop_h6_error": loop6_error,
    }


def run_one(
    mode: str,
    predictor_weight: float,
    alignment_weight: float,
    seed: int,
    config: EquivarianceBenchmarkConfig,
) -> dict[str, Any]:
    model, _, _, _, rollout, _, best_epoch, best_val = _fit_joint(
        mode, predictor_weight, alignment_weight, seed, config
    )
    metrics = _loop_metrics(model, seed)
    metrics["rollout_h8_mse"] = float(_rollout_mse(model, rollout, (8,))["8"])
    return {
        "mode": mode,
        "predictor_weight": float(predictor_weight),
        "alignment_weight": float(alignment_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val),
        **metrics,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1 if array.size > 1 else 0))}


def _pearson(x: list[float], y: list[float]) -> float:
    x_array, y_array = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x_array.size < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return 0.0
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['mode']}:predictor={record['predictor_weight']}:alignment={record['alignment_weight']}"
        grouped.setdefault(key, []).append(record)
    result = {}
    for key, rows in grouped.items():
        summary = {"n_seeds": len(rows)}
        summary.update(
            {
                metric: _mean_std([float(row[metric]) for row in rows])
                for metric in (
                    "inverse_pair_error",
                    "closed_loop_h4_error",
                    "closed_loop_h6_error",
                    "rollout_h8_mse",
                )
            }
        )
        result[key] = summary
    return result


def run_report(seeds: list[int], epochs: int = 120) -> dict[str, Any]:
    config = EquivarianceBenchmarkConfig(epochs=epochs)
    conditions = [
        ("none", 0.0, 0.0),
        ("predictor", 1.0, 0.0),
        ("both", 1.0, 0.003),
    ]
    records = []
    started = time.time()
    for mode, predictor_weight, alignment_weight in conditions:
        for seed in seeds:
            print(f"[inverse-loop] mode={mode} alignment={alignment_weight} seed={seed}")
            records.append(run_one(mode, predictor_weight, alignment_weight, seed, config))
    report = {
        "protocol": {
            "dynamics": "s_next = s + a",
            "inverse_action": "-a",
            "closed_loops": ["a,-a", "a,b,-a,-b", "a,b,c,-a,-b,-c"],
            "conditions": conditions,
            "seeds": [int(seed) for seed in seeds],
            "samples_per_seed": 2048,
            "synthetic_vector_world_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": _summary(records),
        "correlations": {
            metric: _pearson(
                [float(row[metric]) for row in records],
                [float(row["rollout_h8_mse"]) for row in records],
            )
            for metric in ("inverse_pair_error", "closed_loop_h4_error", "closed_loop_h6_error")
        },
        "claim_boundary": [
            "inverse and loop tests use additive synthetic dynamics with explicitly paired inverse actions",
            "correlations are descriptive and do not establish causality",
            "no LeWM, PushT, online planning, or causal-control evidence",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Inverse-action and closed-loop consistency",
        "",
        "The additive vector world provides an explicit inverse action `-a`. Values are mean ± standard deviation over seeds.",
        "",
        "| condition | seeds | inverse pair | closed loop h=4 | closed loop h=6 | rollout h=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        fmt = lambda metric: f"{values[metric]['mean']:.6g} ± {values[metric]['std']:.2g}"
        lines.append(
            f"| {key} | {values['n_seeds']} | {fmt('inverse_pair_error')} | "
            f"{fmt('closed_loop_h4_error')} | {fmt('closed_loop_h6_error')} | {fmt('rollout_h8_mse')} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive correlations with h=8 rollout",
            "",
            *[f"- `{metric}`: Pearson {value:.4f}" for metric, value in report["correlations"].items()],
            "",
            "These are pooled descriptive correlations across condition-seed records, not causal estimates.",
            "",
            "## Boundary",
            "",
            "This is synthetic mechanism evidence only; it does not establish nonlinear LeWM or online planning behavior.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    report = run_report(args.seeds, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
