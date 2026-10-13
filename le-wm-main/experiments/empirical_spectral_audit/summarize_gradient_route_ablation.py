"""Summarize paired gradient-route ablations for the learned pixel world.

The input is the JSON emitted by ``run_pixel_world_model_synthetic.py``.  The
summary keeps route comparisons paired by seed and reports deterministic
bootstrap intervals plus an exact sign-flip reference for the small number of
seeds used in this CPU diagnostic.  It is descriptive evidence, not a claim of
large-sample inferential validity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any


METRICS: dict[str, tuple[str, ...]] = {
    "prediction": ("test_one_step_mse",),
    "rollout": ("rollout_mse", "8"),
    "composition": ("action_composition_error_mean",),
    "global_commutator": ("diagnostics", "global_commutator"),
    "conditional_commutator": ("diagnostics", "conditional_commutator"),
    "stable_rank": ("diagnostics", "stable_rank"),
    "covariance_entropy": ("diagnostics", "covariance_eigenvalue_entropy"),
    "covariance_condition": ("diagnostics", "covariance_condition_number"),
}


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric at {path}: {value}")
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _bootstrap_ci(values: list[float], seed: int = 20261003, draws: int = 20000) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = sorted(
        _mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def _sign_flip_p(values: list[float]) -> float:
    """Exact two-sided sign-flip reference for n <= 12 paired differences."""

    if not values:
        return 1.0
    observed = abs(_mean(values))
    if len(values) > 12:
        return float("nan")
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(_mean([sign * value for sign, value in zip(signs, values)])) >= observed - 1e-15:
            count += 1
    return count / total


def _format_interval(values: list[float]) -> str:
    low, high = _bootstrap_ci(values)
    return f"{_mean(values):.6g} [{low:.6g}, {high:.6g}]"


def _condition_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return str(row["gradient_route"]), str(row["mode"]), float(row["alignment_weight"])


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    records = report["records"]
    by_condition: dict[tuple[str, str, float], dict[int, dict[str, Any]]] = {}
    for row in records:
        key = _condition_key(row)
        seed = int(row["seed"])
        if seed in by_condition.setdefault(key, {}):
            raise ValueError(f"duplicate condition/seed: {key}/{seed}")
        by_condition[key][seed] = row

    route_checks = []
    for key, rows in sorted(by_condition.items()):
        route, mode, weight = key
        encoder_grads = [float(row["alignment_gradient_norms"]["encoder"]) for row in rows.values()]
        predictor_grads = [float(row["alignment_gradient_norms"]["predictor"]) for row in rows.values()]
        if route == "predictor" and max(encoder_grads, default=0.0) > 1e-10:
            raise ValueError(f"predictor route leaked encoder gradient: {key}")
        if route == "encoder" and max(predictor_grads, default=0.0) > 1e-10:
            raise ValueError(f"encoder route leaked predictor gradient: {key}")
        if route == "joint" and not (max(encoder_grads, default=0.0) > 0 and max(predictor_grads, default=0.0) > 0):
            raise ValueError(f"joint route did not reach both parameter groups: {key}")
        route_checks.append({
            "route": route,
            "mode": mode,
            "alignment_weight": weight,
            "n_seeds": len(rows),
            "encoder_gradient_max": max(encoder_grads, default=0.0),
            "predictor_gradient_max": max(predictor_grads, default=0.0),
            "passed": True,
        })

    comparisons = []
    for key, rows in sorted(by_condition.items()):
        route, mode, weight = key
        if route in {"none", "joint"}:
            continue
        reference_key = ("joint", mode, weight)
        if reference_key not in by_condition:
            continue
        common = sorted(set(rows) & set(by_condition[reference_key]))
        for metric, path in METRICS.items():
            deltas = [
                _nested(rows[seed], path) - _nested(by_condition[reference_key][seed], path)
                for seed in common
            ]
            comparisons.append({
                "comparison": "route_minus_joint",
                "route": route,
                "mode": mode,
                "alignment_weight": weight,
                "metric": metric,
                "seeds": common,
                "delta_mean": _mean(deltas),
                "delta_bootstrap_ci95": list(_bootstrap_ci(deltas)),
                "exact_sign_flip_p": _sign_flip_p(deltas),
            })

    return {
        "protocol": {
            "source_protocol": report.get("protocol", {}),
            "paired_by": "seed within identical mode and alignment weight",
            "bootstrap_draws": 20000,
            "bootstrap_seed": 20261003,
            "interpretation": "descriptive paired uncertainty for a small CPU synthetic diagnostic",
        },
        "route_checks": route_checks,
        "comparisons": comparisons,
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Gradient-route ablation summary",
        "",
        "This is a paired, small-seed summary of the learned synthetic pixel-world diagnostic. Negative deltas favor the route over the joint reference for error metrics; commutator and covariance deltas are reported descriptively.",
        "",
        "## Gradient isolation checks",
        "",
        "| route | mode | weight | seeds | max encoder grad | max predictor grad | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["route_checks"]:
        lines.append(
            f"| {row['route']} | {row['mode']} | {row['alignment_weight']:.6g} | {row['n_seeds']} | "
            f"{row['encoder_gradient_max']:.6g} | {row['predictor_gradient_max']:.6g} | PASS |"
        )
    lines.extend([
        "",
        "## Paired route minus joint",
        "",
        "| route | mode | weight | metric | mean delta | bootstrap 95% interval | exact sign-flip p |",
        "| --- | --- | ---: | --- | ---: | --- | ---: |",
    ])
    for row in summary["comparisons"]:
        low, high = row["delta_bootstrap_ci95"]
        lines.append(
            f"| {row['route']} | {row['mode']} | {row['alignment_weight']:.6g} | {row['metric']} | "
            f"{row['delta_mean']:.6g} | [{low:.6g}, {high:.6g}] | {row['exact_sign_flip_p']:.6g} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "The route checks establish implementation-level gradient isolation. The paired intervals are not a claim of population-level significance with five seeds. This synthetic benchmark still does not establish nonlinear LeWM, PushT, causal control, or online planning gains.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    summary = summarize(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.output.with_suffix(".md"))
    print(json.dumps({"route_checks": len(summary["route_checks"]), "comparisons": len(summary["comparisons"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
