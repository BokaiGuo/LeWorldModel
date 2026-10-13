"""Paired inference for the learned pixel nuisance-OOD benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


Metric = tuple[Callable[[dict[str, Any]], float], bool]


def _nested(row: dict[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        value = value[key]
    return float(value)


METRICS: dict[str, Metric] = {
    "id_one_step_mse": (lambda row: _nested(row, "id_one_step_mse"), True),
    "ood_one_step_mse": (lambda row: _nested(row, "ood_one_step_mse"), True),
    "id_rollout_h8_mse": (lambda row: _nested(row, "id_rollout_mse", "8"), True),
    "ood_rollout_h8_mse": (lambda row: _nested(row, "ood_rollout_mse", "8"), True),
    "ood_nuisance_to_action_ratio": (
        lambda row: _nested(row, "ood_tangent_diagnostics", "nuisance_to_action_norm_ratio"),
        True,
    ),
    "ood_predicted_action_tangent_cosine": (
        lambda row: _nested(row, "ood_tangent_diagnostics", "predicted_action_tangent_cosine"),
        False,
    ),
    "global_commutator": (lambda row: _nested(row, "diagnostics", "global_commutator"), True),
    "conditional_commutator": (
        lambda row: _nested(row, "diagnostics", "conditional_commutator"),
        True,
    ),
}


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _condition(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["mode"]), float(row["alignment_weight"])


def _paired(rows: list[dict[str, Any]], target, baseline, metric: str) -> np.ndarray:
    target_rows = {int(row["seed"]): row for row in rows if _condition(row) == target}
    baseline_rows = {int(row["seed"]): row for row in rows if _condition(row) == baseline}
    seeds = sorted(set(target_rows).intersection(baseline_rows))
    if not seeds:
        raise ValueError(f"no paired seeds for target={target}, baseline={baseline}")
    extractor, lower_is_better = METRICS[metric]
    values = []
    for seed in seeds:
        raw = extractor(target_rows[seed]) - extractor(baseline_rows[seed])
        values.append(raw if lower_is_better else -raw)
    return np.asarray(values, dtype=np.float64)


def _sign_flip_pvalue(values: np.ndarray) -> float:
    if values.size > 16:
        raise ValueError("exact sign-flip enumeration is bounded at 16 pairs")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        total += 1
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / total)


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalue))
        adjusted[name] = running
    return adjusted


def _summary(values: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(int(repeats))],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(values.size),
        "mean_oriented_difference_target_minus_baseline": float(values.mean()),
        "std_oriented_difference": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "bootstrap_mean_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip_pvalue(values),
        "paired_oriented_differences": [float(value) for value in values],
        "negative_favors_target": True,
    }


def run_report(input_path: Path, repeats: int = 10000, seed: int = 104729) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    baseline = ("none", 0.0)
    conditions = sorted({_condition(row) for row in rows if _condition(row) != baseline})
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same seed and same generated nuisance-OOD protocol",
            "bootstrap_unit": "paired seed-level oriented differences",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "pvalue": "exact two-sided paired sign-flip; Holm correction within each metric",
            "learned_pixel_encoder": True,
            "synthetic_pixel_nuisance_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "negative oriented differences favor the intervention",
            "intervals and exact tests summarize the observed five paired seeds",
            "nuisance/action tangent ratios are diagnostic probes, not proof of invariance",
            "this does not establish nonlinear LeWM, PushT, online planning, causal control, or GPU performance",
        ],
    }
    raw_pvalues: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for index, condition in enumerate(conditions):
        name = f"{condition[0]}_{condition[1]:g}_vs_none"
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            values = _paired(rows, condition, baseline, metric)
            metrics[metric] = _summary(values, repeats, seed + index * 23 + metric_index)
            raw_pvalues[metric][name] = metrics[metric]["exact_two_sided_sign_flip_p"]
        report["comparisons"][name] = {
            "target": {"mode": condition[0], "alignment_weight": condition[1]},
            "baseline": {"mode": baseline[0], "alignment_weight": baseline[1]},
            "metrics": metrics,
        }
    for name, values in report["comparisons"].items():
        for metric in METRICS:
            values["metrics"][metric]["holm_p_within_metric"] = _holm(raw_pvalues[metric])[name]
    if not _finite(report):
        raise ValueError("pixel nuisance inference report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Paired inference for the learned pixel nuisance-OOD benchmark",
        "",
        "Metrics are oriented so negative differences favor the intervention; action-tangent cosine is sign-reversed because higher is better.",
        "",
        "| comparison | metric | n | mean oriented difference | 95% bootstrap CI | exact sign-flip p | Holm p |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {name} | {metric} | {summary['n_paired_seeds']} | {summary['mean_oriented_difference_target_minus_baseline']:.6g} | "
                f"[{ci[0]:.6g}, {ci[1]:.6g}] | {summary['exact_two_sided_sign_flip_p']:.6g} | "
                f"{summary['holm_p_within_metric']:.6g} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=104729)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
