"""Exact action-sequence composition audit for the learned alignment benchmark.

The existing alignment benchmark reports ordered two-action products and the
functional follow-up reports random-sequence rollout.  This experiment closes
the intervening gap by enumerating every binary action sequence of length 2, 4,
and 8 and comparing the learned operator product with the corresponding
data-generating product.  For a sequence ``(a_1, ..., a_h)`` the product is
``A_{a_h} ... A_{a_1}``; no unverified group-composition law is assumed.

The same trained model and deterministic protocol are used to measure
one-step error and random-sequence rollout.  Correlations between exact
operator-product error and rollout error are descriptive per-seed diagnostics,
not causal or nonlinear-LeWM claims.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch

try:
    from experiments.empirical_spectral_audit.run_alignment_functional_synthetic import (
        FunctionalAlignmentConfig,
        _fit_model,
        _random_rollout_mse,
        _sample_sequence_split,
    )
    from experiments.empirical_spectral_audit.run_alignment_training_synthetic import (
        _make_ground_truth,
    )
except ModuleNotFoundError:  # standalone reproducibility-package execution
    from run_alignment_functional_synthetic import (
        FunctionalAlignmentConfig,
        _fit_model,
        _random_rollout_mse,
        _sample_sequence_split,
    )
    from run_alignment_training_synthetic import _make_ground_truth


COMPOSITION_HORIZONS = (2, 4, 8)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@torch.no_grad()
def _sequence_product(operators: torch.Tensor, sequence: Iterable[int]) -> torch.Tensor:
    """Return A_last ... A_first for an ordered action sequence."""

    product = torch.eye(operators.size(-1), dtype=operators.dtype, device=operators.device)
    for action in sequence:
        product = operators[int(action)] @ product
    return product


@torch.no_grad()
def _sequence_composition_error(
    learned_operators: torch.Tensor,
    true_operators: torch.Tensor,
    horizon: int,
) -> dict[str, float]:
    """Enumerate all binary sequences and compare learned/true products."""

    if learned_operators.shape != true_operators.shape:
        raise ValueError("learned and true operators must have identical shape")
    if learned_operators.ndim != 3 or learned_operators.size(0) != 2:
        raise ValueError("this audit expects exactly two action operators")

    errors: list[float] = []
    for sequence in itertools.product(range(2), repeat=horizon):
        learned = _sequence_product(learned_operators, sequence)
        truth = _sequence_product(true_operators, sequence)
        denominator = truth.square().sum().sqrt().clamp_min(1e-12)
        errors.append(float(((learned - truth).square().sum().sqrt() / denominator).cpu()))
    mean = sum(errors) / len(errors)
    std = 0.0 if len(errors) < 2 else math.sqrt(
        sum((value - mean) ** 2 for value in errors) / (len(errors) - 1)
    )
    return {
        "n_sequences": float(len(errors)),
        "mean": mean,
        "std": std,
        "p90": _percentile(errors, 0.90),
        "max": max(errors),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires paired vectors of length at least two")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denominator = math.sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    return 0.0 if denominator == 0.0 else numerator / denominator


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _correlation(xs: list[float], ys: list[float]) -> dict[str, float]:
    return {"pearson": _pearson(xs, ys), "spearman": _pearson(_rank(xs), _rank(ys))}


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def run_one(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: FunctionalAlignmentConfig,
) -> dict[str, object]:
    bundle = _fit_model(mode, alignment_weight, seed, config)
    true_covariances, true_operators = _make_ground_truth(4, torch.device("cpu"))
    del true_covariances
    generator = torch.Generator().manual_seed(100_000 + seed)
    id_rollout = _sample_sequence_split(
        config.rollout_trajectories,
        config.rollout_horizon,
        *_make_ground_truth(4, torch.device("cpu")),
        generator,
    )
    ood_rollout = _sample_sequence_split(
        config.rollout_trajectories,
        config.rollout_horizon,
        *_make_ground_truth(4, torch.device("cpu")),
        generator,
        group_one_probability=config.ood_group_one_probability,
        state_shift=config.ood_state_shift,
    )
    with torch.no_grad():
        one_step = (
            bundle.model(bundle.test["source"], bundle.test["action"])
            - bundle.test["target"]
        ).square().mean()
        sequence_errors = {
            str(horizon): _sequence_composition_error(
                bundle.model.operators, true_operators, horizon
            )
            for horizon in COMPOSITION_HORIZONS
        }
    record: dict[str, object] = {
        "mode": mode,
        "alignment_weight": alignment_weight,
        "seed": seed,
        "device": "cpu",
        "best_epoch": bundle.best_epoch,
        "best_validation_mse": bundle.best_validation_mse,
        "test_one_step_mse": float(one_step),
        "sequence_composition_error": sequence_errors,
        "random_rollout_mse": {
            "id": _random_rollout_mse(bundle.model, id_rollout, true_operators, (2, 4, 8)),
            "ood": _random_rollout_mse(bundle.model, ood_rollout, true_operators, (2, 4, 8)),
        },
        "protocol": {
            "composition_definition": "A_learned_last...A_learned_first versus A_true_last...A_true_first",
            "composition_sequence_family": "all binary action sequences",
            "composition_horizons": list(COMPOSITION_HORIZONS),
            "rollout_action_schedule": "random_group_sequence",
            "synthetic_only": True,
        },
    }
    if not _finite(record):
        raise FloatingPointError(f"non-finite record: {mode}:{alignment_weight}:{seed}")
    return record


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(f"{record['mode']}:{record['alignment_weight']}", []).append(record)

    def mean_std(values: list[float]) -> dict[str, float]:
        mean = sum(values) / len(values)
        std = 0.0 if len(values) < 2 else math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        return {"mean": mean, "std": std}

    output: dict[str, object] = {}
    for key, rows in groups.items():
        summary: dict[str, object] = {"n_seeds": len(rows)}
        summary["test_one_step_mse"] = mean_std([float(row["test_one_step_mse"]) for row in rows])
        for split in ("id", "ood"):
            for horizon in COMPOSITION_HORIZONS:
                summary[f"{split}_rollout_h{horizon}_mse"] = mean_std([
                    float(row["random_rollout_mse"][split][str(horizon)])  # type: ignore[index]
                    for row in rows
                ])
        for horizon in COMPOSITION_HORIZONS:
            for statistic in ("mean", "p90", "max"):
                summary[f"composition_h{horizon}_{statistic}"] = mean_std([
                    float(row["sequence_composition_error"][str(horizon)][statistic])  # type: ignore[index]
                    for row in rows
                ])
        output[key] = summary
    return output


def _correlation_block(
    records: list[dict[str, object]],
    transform=lambda value: value,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for horizon in COMPOSITION_HORIZONS:
        composition = [transform(float(row["sequence_composition_error"][str(horizon)]["mean"])) for row in records]  # type: ignore[index]
        output[f"composition_h{horizon}_vs_rollout"] = {
            split: _correlation(
                composition,
                [transform(float(row["random_rollout_mse"][split][str(horizon)])) for row in records],  # type: ignore[index]
            )
            for split in ("id", "ood")
        }
    by_seed: dict[str, list[dict[str, object]]] = {}
    for row in records:
        by_seed.setdefault(str(row["seed"]), []).append(row)
    return output


def _correlations(records: list[dict[str, object]]) -> dict[str, object]:
    """Compute pooled and per-seed descriptive correlations without recursion."""

    by_seed: dict[str, list[dict[str, object]]] = {}
    for row in records:
        by_seed.setdefault(str(row["seed"]), []).append(row)
    by_mode: dict[str, list[dict[str, object]]] = {}
    for row in records:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    return {
        **_correlation_block(records),
        "log10_pooled": _correlation_block(
            records, transform=lambda value: math.log10(max(value, 1e-12))
        ),
        "per_mode": {
            mode: _correlation_block(rows) for mode, rows in sorted(by_mode.items())
        },
        "per_seed": {
            seed: _correlation_block(rows) for seed, rows in sorted(by_seed.items())
        },
    }


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# Exact action-sequence composition audit",
        "",
        "This is a CPU-only synthetic action-conditioned benchmark. It enumerates all binary action sequences of length 2, 4, and 8 and compares learned operator products with the corresponding data-generating products; it is not nonlinear LeWM, PushT, online control, or GPU evidence.",
        "",
        "For `(a_1, ..., a_h)`, the product is `A_{a_h} ... A_{a_1}`. No group-composition law is assumed; the metric is a direct product comparison in the controlled linear world.",
        "",
        "| mode | weight | seeds | comp h=2 mean | comp h=4 mean | comp h=8 mean | comp h=8 P90 | ID rollout h=8 | OOD rollout h=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        def fmt(name: str) -> str:
            value = values[name]
            return f"{value['mean']:.6g} +/- {value['std']:.2g}"
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | "
            f"{fmt('composition_h2_mean')} | {fmt('composition_h4_mean')} | "
            f"{fmt('composition_h8_mean')} | {fmt('composition_h8_p90')} | "
            f"{fmt('id_rollout_h8_mse')} | {fmt('ood_rollout_h8_mse')} |"
        )
    lines.extend(["", "## Composition--rollout association", ""])
    lines.append("The following are descriptive correlations across the 33 per-seed records; they are not inferential or causal estimates.")
    lines.append("")
    lines.append("| composition horizon | rollout split | Pearson | Spearman |")
    lines.append("| ---: | --- | ---: | ---: |")
    for horizon in COMPOSITION_HORIZONS:
        for split in ("id", "ood"):
            values = report["correlations"][f"composition_h{horizon}_vs_rollout"][split]  # type: ignore[index]
            lines.append(f"| {horizon} | {split} | {values['pearson']:.4f} | {values['spearman']:.4f} |")
    lines.extend(["", "The h=8 Pearson association remains positive within each intervention family: "])
    for mode, values in report["correlations"]["per_mode"].items():  # type: ignore[index]
        id_value = values["composition_h8_vs_rollout"]["id"]["pearson"]
        ood_value = values["composition_h8_vs_rollout"]["ood"]["pearson"]
        lines.append(f"- `{mode}`: ID {id_value:.4f}; OOD {ood_value:.4f}.")
    lines.append("The `log10_pooled` block in the JSON provides a scale-robust sensitivity check; all correlations are descriptive and no significance test is claimed.")
    lines.extend([
        "",
        "Interpretation boundary: sequence-product error is a structural diagnostic in the controlled linear benchmark. Its association with rollout error does not establish that alignment causes nonlinear world-model or planning improvements.",
    ])
    output.write_text("\n".join(lines) + "\n")


def _write_figure(report: dict[str, object], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return
    records = report["records"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    colors = {"none": "#555555", "global": "#c44e52", "conditional": "#4c72b0"}
    for split, axis in zip(("id", "ood"), axes):
        for row in records:
            x = row["sequence_composition_error"]["8"]["mean"]
            y = row["random_rollout_mse"][split]["8"]
            axis.scatter(x, y, s=24, alpha=0.78, color=colors[row["mode"]])
        axis.set_xlabel("Exact sequence-product error (h=8)")
        axis.set_ylabel(f"Random-sequence rollout MSE (h=8, {split.upper()})")
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.grid(True, alpha=0.25)
    fig.suptitle("Action composition error and long-horizon rollout")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=mode.title())
        for mode, color in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    args = parser.parse_args()
    config = FunctionalAlignmentConfig(epochs=args.epochs)
    records: list[dict[str, object]] = []
    start = time.time()
    for mode in ("none", "global", "conditional"):
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(
                    f"[sequence-composition] mode={mode} weight={weight} seed={seed}: "
                    "enumerating h=2/4/8 action products and measuring rollout"
                )
                records.append(run_one(mode, weight, seed, config))
    report = {
        "protocol": {
            "device": "cpu",
            "seeds": args.seeds,
            "modes": ["none", "global", "conditional"],
            "weights": args.weights,
            "epochs": args.epochs,
            "composition_horizons": list(COMPOSITION_HORIZONS),
            "sequence_count_by_horizon": {str(horizon): 2 ** horizon for horizon in COMPOSITION_HORIZONS},
            "learned_training": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - start,
        },
        "records": records,
        "summary": _summary(records),
        "correlations": _correlations(records),
    }
    if not _finite(report):
        raise FloatingPointError("non-finite report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    _write_figure(report, args.output.with_name("composition_rollout_scatter.png"))
    print(json.dumps(report["correlations"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
