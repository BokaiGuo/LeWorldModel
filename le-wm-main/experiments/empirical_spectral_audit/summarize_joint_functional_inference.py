"""Paired inference for the factorial functional ablation.

The interaction is computed on an oriented metric scale, so a negative value
means that the joint intervention is better than the additive main-effect
prediction.  This remains a seed-level synthetic inference report.
"""

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
    "test_one_step_mse": (lambda row: _nested(row, "test_one_step_mse"), True),
    "id_rollout_h8_mse": (lambda row: _nested(row, "id_rollout_mse", "8"), True),
    "ood_action_rollout_h8_mse": (
        lambda row: _nested(row, "ood_rollout_mse", "action_2x", "8"),
        True,
    ),
    "ood_combined_rollout_h8_mse": (
        lambda row: _nested(row, "ood_rollout_mse", "combined", "8"),
        True,
    ),
    "ood_planning_regret": (
        lambda row: _nested(row, "planning", "ood_combined", "planning_regret"),
        True,
    ),
    "ood_planning_success": (
        lambda row: _nested(row, "planning", "ood_combined", "goal_success_rate"),
        False,
    ),
    "predictor_equivariance_error": (
        lambda row: _nested(row, "diagnostics", "predictor_equivariance_error"),
        True,
    ),
    "operator_alignment": (
        lambda row: _nested(row, "diagnostics", "operator_alignment"),
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


def _condition(row: dict[str, Any]) -> str:
    return str(row["mode"])


def _oriented_value(row: dict[str, Any], metric: str) -> float:
    extractor, lower_is_better = METRICS[metric]
    value = extractor(row)
    return float(value if lower_is_better else -value)


def _rows_by_condition(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(_condition(row), {})[int(row["seed"])] = row
    return output


def _paired(rows: list[dict[str, Any]], target: str, baseline: str, metric: str) -> np.ndarray:
    grouped = _rows_by_condition(rows)
    target_rows = grouped[target]
    baseline_rows = grouped[baseline]
    seeds = sorted(set(target_rows).intersection(baseline_rows))
    if not seeds:
        raise ValueError(f"no paired seeds for target={target}, baseline={baseline}")
    return np.asarray(
        [_oriented_value(target_rows[seed], metric) - _oriented_value(baseline_rows[seed], metric) for seed in seeds],
        dtype=np.float64,
    )


def _interaction(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    grouped = _rows_by_condition(rows)
    required = {"none", "predictor", "alignment", "both"}
    if not required.issubset(grouped):
        raise ValueError(f"missing factorial conditions: {required - set(grouped)}")
    seeds = sorted(set.intersection(*(set(grouped[name]) for name in required)))
    if not seeds:
        raise ValueError("no paired seeds for factorial interaction")
    return np.asarray(
        [
            _oriented_value(grouped["both"][seed], metric)
            - _oriented_value(grouped["predictor"][seed], metric)
            - _oriented_value(grouped["alignment"][seed], metric)
            + _oriented_value(grouped["none"][seed], metric)
            for seed in seeds
        ],
        dtype=np.float64,
    )


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
        "mean_oriented_difference": float(values.mean()),
        "std_oriented_difference": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "bootstrap_mean_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip_pvalue(values),
        "paired_oriented_differences": [float(value) for value in values],
        "negative_favors_target": True,
    }


def run_report(input_path: Path, repeats: int = 10000, seed: int = 161803) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    comparisons = {
        "predictor_vs_none": ("predictor", "none"),
        "alignment_vs_none": ("alignment", "none"),
        "both_vs_none": ("both", "none"),
        "both_vs_predictor": ("both", "predictor"),
        "interaction_both_minus_predictor_minus_alignment_plus_none": None,
    }
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "paired_unit": "same seed and same factorial protocol",
            "bootstrap_unit": "paired seed-level oriented differences",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "interaction": "oriented(both) - oriented(predictor) - oriented(alignment) + oriented(none)",
            "pvalue": "exact two-sided paired sign-flip; Holm correction within each metric",
            "synthetic_functional_proxy_only": True,
        },
        "comparisons": {},
        "claim_boundary": [
            "negative oriented differences favor the target comparison",
            "negative interaction means the joint condition is better than the additive main-effect prediction on the oriented metric",
            "intervals and exact tests summarize the observed ten paired seeds",
            "the planning metric is a candidate-set proxy, not online task success or causal control",
        ],
    }
    raw_pvalues: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    for index, (name, comparison) in enumerate(comparisons.items()):
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            if comparison is None:
                values = _interaction(rows, metric)
            else:
                values = _paired(rows, comparison[0], comparison[1], metric)
            metrics[metric] = _summary(values, repeats, seed + index * 37 + metric_index)
            raw_pvalues[metric][name] = metrics[metric]["exact_two_sided_sign_flip_p"]
        report["comparisons"][name] = {
            "comparison": comparison,
            "metrics": metrics,
        }
    for name, values in report["comparisons"].items():
        for metric in METRICS:
            values["metrics"][metric]["holm_p_within_metric"] = _holm(raw_pvalues[metric])[name]
    if not _finite(report):
        raise ValueError("joint functional inference report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Paired inference for the factorial functional ablation",
        "",
        "Metrics are oriented so negative differences favor the target. The interaction is `(both - predictor) - (alignment - none)` on that oriented scale; negative interaction indicates a joint benefit beyond the additive main-effect prediction.",
        "",
        "| comparison | metric | n | mean oriented difference | 95% bootstrap CI | exact sign-flip p | Holm p |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, values in report["comparisons"].items():
        for metric, summary in values["metrics"].items():
            ci = summary["bootstrap_mean_ci95"]
            lines.append(
                f"| {name} | {metric} | {summary['n_paired_seeds']} | {summary['mean_oriented_difference']:.6g} | "
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
    parser.add_argument("--bootstrap-seed", type=int, default=161803)
    args = parser.parse_args()
    report = run_report(args.input, args.bootstrap_repeats, args.bootstrap_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
