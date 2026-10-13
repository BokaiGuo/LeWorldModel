"""Matched one-step-error inference for the synthetic functional benchmarks.

The existing functional runs are paired by seed, but an intervention can also
change one-step prediction error.  This report therefore performs a prespecified
secondary analysis: for every target row, select the nearest baseline row with
the same seed in one-step error, retain the pair only when its relative error
gap is within a declared tolerance, and evaluate downstream outcomes on the
retained pairs.  Keeping the seed fixed prevents a cross-seed match from
silently changing the data-generating protocol.

This is a sensitivity analysis of synthetic/vector-world proxies.  It does not
turn the candidate-set planning metric into online control evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


Metric = tuple[Callable[[dict[str, Any]], float], bool]


def _nested(row: dict[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        value = value[key]
    return float(value)


def _mean_dict(row: dict[str, Any], key: str) -> float:
    values = [float(value) for value in row[key].values()]
    return float(np.mean(values))


def _metrics_for(kind: str) -> dict[str, Metric]:
    if kind == "alignment":
        return {
            "test_one_step_mse": (lambda r: _nested(r, "test_one_step_mse"), True),
            "id_rollout_h8_mse": (lambda r: _nested(r, "random_rollout_mse", "id", "8"), True),
            "ood_rollout_h8_mse": (lambda r: _nested(r, "random_rollout_mse", "ood", "8"), True),
            "composition_error": (lambda r: _mean_dict(r, "action_composition_error"), True),
            "id_planning_regret": (lambda r: _nested(r, "planning", "id", "planning_regret"), True),
            "ood_planning_regret": (lambda r: _nested(r, "planning", "ood", "planning_regret"), True),
            "id_planning_success": (lambda r: _nested(r, "planning", "id", "goal_success_rate"), False),
            "ood_planning_success": (lambda r: _nested(r, "planning", "ood", "goal_success_rate"), False),
            "global_commutator": (lambda r: _nested(r, "diagnostics", "global_commutator"), True),
            "conditional_commutator": (
                lambda r: _nested(r, "diagnostics", "conditional_commutator_mean"),
                True,
            ),
        }
    if kind == "nonlinear_local":
        return {
            "one_step_mse": (lambda r: _nested(r, "test_one_step_mse"), True),
            "id_rollout_h8_mse": (lambda r: _nested(r, "id_rollout_mse", "8"), True),
            "ood_state_shift_rollout_h8_mse": (
                lambda r: _nested(r, "ood_rollout_mse", "state_shift", "8"), True
            ),
            "ood_action_imbalance_rollout_h8_mse": (
                lambda r: _nested(r, "ood_rollout_mse", "action_imbalance", "8"), True
            ),
            "state_shift_composition_error": (
                lambda r: _nested(r, "composition_error", "state_shift"), True
            ),
            "ood_planning_regret": (
                lambda r: _nested(r, "planning", "state_shift", "planning_regret"), True
            ),
            "ood_planning_success": (
                lambda r: _nested(r, "planning", "state_shift", "goal_success_rate"), False
            ),
            "global_local_commutator": (
                lambda r: _nested(r, "diagnostics", "global_local_commutator"), True
            ),
            "conditional_local_commutator": (
                lambda r: _nested(r, "diagnostics", "conditional_local_commutator"), True
            ),
            "predictor_equivariance_error": (
                lambda r: _nested(r, "diagnostics", "predictor_equivariance_error"), True
            ),
        }
    if kind == "joint":
        one_step_key = "test_one_step_mse" if kind == "joint" else "train_one_step_mse"
        return {
            "one_step_mse": (lambda r, k=one_step_key: _nested(r, k), True),
            "id_rollout_h8_mse": (lambda r: _nested(r, "id_rollout_mse", "8"), True),
            "ood_action_rollout_h8_mse": (lambda r: _nested(r, "ood_rollout_mse", "action_2x", "8"), True),
            "ood_combined_rollout_h8_mse": (lambda r: _nested(r, "ood_rollout_mse", "combined", "8"), True),
            "in_distribution_composition_error": (
                lambda r: _nested(r, "composition_error", "in_distribution"), True
            ),
            "ood_composition_error": (lambda r: _nested(r, "composition_error", "ood_action"), True),
            "ood_planning_regret": (
                lambda r: _nested(r, "planning", "ood_combined", "planning_regret"), True
            ),
            "ood_planning_success": (
                lambda r: _nested(r, "planning", "ood_combined", "goal_success_rate"), False
            ),
        }
    if kind == "equivariance":
        return {
            "one_step_mse": (lambda r: _nested(r, "train_one_step_mse"), True),
            "ood_action_rollout_h8_mse": (lambda r: _nested(r, "ood_rollout_mse", "action_2x", "8"), True),
            "ood_combined_rollout_h8_mse": (lambda r: _nested(r, "ood_rollout_mse", "combined", "8"), True),
            "in_distribution_composition_error": (
                lambda r: _nested(r, "composition_error", "in_distribution"), True
            ),
            "ood_composition_error": (lambda r: _nested(r, "composition_error", "ood_action"), True),
            "ood_planning_regret": (
                lambda r: _nested(r, "planning", "ood_combined", "planning_regret"), True
            ),
            "ood_planning_success": (
                lambda r: _nested(r, "planning", "ood_combined", "goal_success_rate"), False
            ),
        }
    raise ValueError(f"unknown benchmark kind: {kind}")


def _kind_from_path(path: Path) -> str:
    name = path.name.lower()
    if "alignment_functional" in name:
        return "alignment"
    if "joint_functional" in name:
        return "joint"
    if "equivariance_functional" in name:
        return "equivariance"
    if "nonlinear_local" in name:
        return "nonlinear_local"
    raise ValueError(f"cannot infer benchmark kind from filename: {path}")


def _condition(row: dict[str, Any], kind: str) -> str:
    if kind == "alignment":
        return f"{row['mode']}@{float(row['alignment_weight']):g}"
    if kind == "equivariance":
        return f"{row['mode']}@{float(row['regularization_weight']):g}"
    return str(row["mode"])


def _condition_sort_key(name: str) -> tuple[str, float]:
    mode, weight = name.rsplit("@", 1) if "@" in name else (name, "0")
    return mode, float(weight)


def _one_step(row: dict[str, Any], kind: str) -> float:
    key = "test_one_step_mse" if kind in {"alignment", "joint", "nonlinear_local"} else "train_one_step_mse"
    value = float(row[key])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"invalid one-step error: {value}")
    return value


@dataclass(frozen=True)
class Pair:
    seed: int
    target: dict[str, Any]
    baseline: dict[str, Any]
    relative_one_step_gap: float


def _relative_gap(target: float, baseline: float) -> float:
    scale = max(abs(float(baseline)), 1e-12)
    return abs(float(target) - float(baseline)) / scale


def _nearest_same_seed_pairs(
    rows: list[dict[str, Any]], kind: str, target_name: str, baseline_name: str
) -> list[Pair]:
    targets: dict[int, list[dict[str, Any]]] = {}
    baselines: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        name = _condition(row, kind)
        bucket = targets if name == target_name else baselines if name == baseline_name else None
        if bucket is not None:
            bucket.setdefault(int(row["seed"]), []).append(row)
    pairs: list[Pair] = []
    for seed in sorted(set(targets).intersection(baselines)):
        for target in targets[seed]:
            baseline = min(
                baselines[seed],
                key=lambda row: abs(_one_step(target, kind) - _one_step(row, kind)),
            )
            pairs.append(
                Pair(
                    seed=seed,
                    target=target,
                    baseline=baseline,
                    relative_one_step_gap=_relative_gap(
                        _one_step(target, kind), _one_step(baseline, kind)
                    ),
                )
            )
    return pairs


def _sign_flip_pvalue(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
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


def _bootstrap_summary(values: np.ndarray, repeats: int, seed: int) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n_paired_seeds": 0,
            "mean_oriented_difference": None,
            "std_oriented_difference": None,
            "bootstrap_mean_ci95": None,
            "exact_two_sided_sign_flip_p": None,
            "paired_oriented_differences": [],
            "evidence_status": "insufficient_matched_pairs",
        }
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
        "evidence_status": "matched_inference" if values.size >= 3 else "exploratory_fewer_than_3_pairs",
    }


def _holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((name, value) for name, value in pvalues.items() if value is not None), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in pvalues}
    running = 0.0
    for rank, (name, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def _oriented_difference(pair: Pair, metric: Metric) -> float:
    extractor, lower_is_better = metric
    target = extractor(pair.target)
    baseline = extractor(pair.baseline)
    if not np.isfinite(target) or not np.isfinite(baseline):
        raise ValueError("non-finite downstream metric in matched pair")
    raw = target - baseline
    return float(raw if lower_is_better else -raw)


def _comparisons(kind: str, conditions: list[str]) -> list[tuple[str, str, str]]:
    baseline = "none@0" if kind not in {"joint", "nonlinear_local"} else "none"
    nonbaseline = [name for name in conditions if name != baseline]
    comparisons = [(f"{name}_vs_{baseline}", name, baseline) for name in nonbaseline]
    if kind == "alignment":
        for weight in sorted({name.split("@", 1)[1] for name in conditions if name.startswith("global@")}):
            target = f"conditional@{weight}"
            base = f"global@{weight}"
            if target in conditions and base in conditions:
                comparisons.append((f"{target}_vs_{base}", target, base))
    elif kind == "equivariance":
        for weight in sorted({name.split("@", 1)[1] for name in conditions if name.startswith("predictor@")}):
            for target, base in [(f"predictor@{weight}", f"latent@{weight}"), (f"both@{weight}", f"predictor@{weight}")]:
                if target in conditions and base in conditions:
                    comparisons.append((f"{target}_vs_{base}", target, base))
    elif kind in {"joint", "nonlinear_local"} and "both" in conditions and "predictor" in conditions:
        comparisons.append(("both_vs_predictor", "both", "predictor"))
    return comparisons


def run_report(
    input_path: Path, tolerances: Iterable[float] = (0.10, 0.25, 0.50), repeats: int = 10000, seed: int = 20261003
) -> dict[str, Any]:
    tolerances = tuple(float(tolerance) for tolerance in tolerances)
    source = json.loads(input_path.read_text())
    rows = list(source["records"])
    kind = _kind_from_path(input_path)
    metrics = _metrics_for(kind)
    conditions = sorted({_condition(row, kind) for row in rows}, key=_condition_sort_key)
    comparisons = _comparisons(kind, conditions)
    report: dict[str, Any] = {
        "protocol": {
            "input": str(input_path),
            "benchmark_kind": kind,
            "matching_unit": "same seed; nearest baseline row in one-step error within that seed",
            "one_step_error": "test_one_step_mse" if kind in {"alignment", "joint", "nonlinear_local"} else "train_one_step_mse",
            "relative_gap": "abs(target-baseline)/max(abs(baseline), 1e-12)",
            "tolerances": [float(tolerance) for tolerance in tolerances],
            "no_cross_seed_matching": True,
            "bootstrap_unit": "retained same-seed matched pairs",
            "bootstrap_repeats": int(repeats),
            "bootstrap_seed": int(seed),
            "pvalue": "exact two-sided paired sign-flip; global Holm correction across all reported comparison-metric tests within each tolerance",
            "synthetic_functional_proxy_only": True,
        },
        "conditions": conditions,
        "comparisons": {},
        "claim_boundary": [
            "matching filters on observed one-step error and does not equalize unobserved capacity or optimization state",
            "a small matched set is reported as insufficient or exploratory, not as confirmatory evidence",
            "negative oriented differences favor the target for every metric; success rates are sign-reversed",
            "planning is a candidate-set proxy, not online task success or causal control",
            "these results do not establish nonlinear LeWorldModel, PushT retraining, GPU, or real-world performance",
        ],
    }
    raw_pvalues_by_tolerance: dict[str, dict[str, float | None]] = {
        f"{float(tolerance):g}": {} for tolerance in tolerances
    }
    for comparison_index, (name, target_name, baseline_name) in enumerate(comparisons):
        pairs = _nearest_same_seed_pairs(rows, kind, target_name, baseline_name)
        comparison_report: dict[str, Any] = {
            "target": target_name,
            "baseline": baseline_name,
            "candidate_pairs": int(len(pairs)),
            "matched_by_tolerance": {},
        }
        for tolerance_index, tolerance in enumerate(tolerances):
            tolerance = float(tolerance)
            selected = [pair for pair in pairs if pair.relative_one_step_gap <= tolerance]
            metrics_report: dict[str, Any] = {}
            for metric_index, (metric_name, metric) in enumerate(metrics.items()):
                values = np.asarray([_oriented_difference(pair, metric) for pair in selected], dtype=np.float64)
                summary = _bootstrap_summary(values, repeats, seed + comparison_index * 997 + tolerance_index * 101 + metric_index)
                metrics_report[metric_name] = summary
                raw_pvalues_by_tolerance[f"{tolerance:g}"][f"{name}::{metric_name}"] = summary[
                    "exact_two_sided_sign_flip_p"
                ]
            comparison_report["matched_by_tolerance"][f"{tolerance:g}"] = {
                "relative_tolerance": tolerance,
                "n_matched": int(len(selected)),
                "match_rate_among_common_seed_candidates": float(len(selected) / len(pairs)) if pairs else 0.0,
                "matched_seeds": [int(pair.seed) for pair in selected],
                "relative_one_step_gaps": [float(pair.relative_one_step_gap) for pair in selected],
                "mean_relative_one_step_gap": float(np.mean([pair.relative_one_step_gap for pair in selected])) if selected else None,
                "metrics": metrics_report,
            }
        report["comparisons"][name] = comparison_report
    for tolerance_key, pvalues in raw_pvalues_by_tolerance.items():
        holm = _holm(pvalues)
        for name, comparison in report["comparisons"].items():
            result = comparison["matched_by_tolerance"][tolerance_key]
            for metric_name, summary in result["metrics"].items():
                summary["holm_p_across_all_comparisons_and_metrics"] = holm[
                    f"{name}::{metric_name}"
                ]
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Matched one-step-error inference for the functional benchmark",
        "",
        "Pairs keep the same seed and select the nearest baseline row in one-step error within that seed. Rows are retained only when the relative gap is within the displayed tolerance. Metrics are oriented so negative differences favor the target.",
        "",
        "| comparison | tolerance | n | match rate | mean one-step gap | metric | mean oriented difference | 95% bootstrap CI | exact sign-flip p | Holm p | status |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for name, comparison in report["comparisons"].items():
        for tolerance, result in comparison["matched_by_tolerance"].items():
            for metric, summary in result["metrics"].items():
                ci = summary["bootstrap_mean_ci95"]
                ci_text = "—" if ci is None else f"[{ci[0]:.6g}, {ci[1]:.6g}]"
                mean_text = "—" if summary["mean_oriented_difference"] is None else f"{summary['mean_oriented_difference']:.6g}"
                p_text = "—" if summary["exact_two_sided_sign_flip_p"] is None else f"{summary['exact_two_sided_sign_flip_p']:.6g}"
                holm_text = "—" if summary.get("holm_p_across_all_comparisons_and_metrics") is None else f"{summary['holm_p_across_all_comparisons_and_metrics']:.6g}"
                gap = "—" if result["mean_relative_one_step_gap"] is None else f"{result['mean_relative_one_step_gap']:.6g}"
                lines.append(
                    f"| {name} | {tolerance} | {result['n_matched']} | {result['match_rate_among_common_seed_candidates']:.3f} | {gap} | {metric} | {mean_text} | {ci_text} | {p_text} | {holm_text} | {summary['evidence_status']} |"
                )
    lines.extend(["", "## Protocol and boundary", ""])
    lines.extend(f"- {item}" for item in report["claim_boundary"])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="functional JSON; may be repeated")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, action="append", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    args = parser.parse_args()
    tolerances = tuple(args.tolerance or (0.10, 0.25, 0.50))
    if any(tolerance <= 0 for tolerance in tolerances):
        raise ValueError("matching tolerances must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"reports": []}
    for input_path in args.input:
        report = run_report(input_path, tolerances, args.bootstrap_repeats, args.bootstrap_seed)
        kind = report["protocol"]["benchmark_kind"]
        json_path = args.output_dir / f"matched_{kind}_inference.json"
        md_path = json_path.with_suffix(".md")
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        _write_markdown(report, md_path)
        manifest["reports"].append({"kind": kind, "json": str(json_path), "markdown": str(md_path)})
        print(f"wrote {json_path}")
    (args.output_dir / "README.md").write_text(
        "# Matched one-step-error functional inference\n\n"
        "This directory contains same-seed nearest-neighbor matching on one-step error for the synthetic functional benchmarks. "
        "See each benchmark report for tolerance-specific downstream inference and claim boundaries.\n\n"
        + "\n".join(f"- `{item['kind']}`: `{item['json']}` / `{item['markdown']}`" for item in manifest["reports"])
        + "\n"
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
