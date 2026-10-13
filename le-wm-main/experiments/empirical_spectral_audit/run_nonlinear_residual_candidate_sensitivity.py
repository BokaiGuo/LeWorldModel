"""Candidate-set robustness audit for residual-aware planner geometry.

This is a focused companion to the strength/calibration audit.  It keeps the
matched-spectrum random-basis control and independent residual calibration,
but varies candidate-set size (16/48/96).  The question is whether the
aligned-minus-random residual signal survives a change in search-set density.
It remains a synthetic candidate-ranking diagnostic, not online LeWM or
PushT control.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        _fit_model,
        _sample_split,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
        SHIFTS,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_residual_strength_calibration_audit import (
        _residual_precision,
        _strength_matrices,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model, _sample_split
    from run_nonlinear_planner_metric_ablation import SHIFTS, _evaluate_metric, _make_task_batch, _predicted_endpoints
    from run_nonlinear_residual_strength_calibration_audit import _residual_precision, _strength_matrices


MODEL_CONDITIONS = ("none", "both")
SOURCES = ("in_sample", "heldout_calibration")
GEOMETRIES = ("aligned", "random_basis")
STRENGTHS = (0.5, 1.0)
CANDIDATE_COUNTS = (16, 48, 96)


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / (2 ** int(values.size)))


def _paired_summary(values: np.ndarray, seed: int, label: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(10000)],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(values.size),
        f"mean_delta_{label}": float(values.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
        "n_negative_deltas": int(np.sum(values < 0.0)),
        "negative_delta_fraction": float(np.mean(values < 0.0)),
        "paired_deltas": [float(value) for value in values],
    }


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted


def _metric_names() -> list[str]:
    return ["euclidean"] + [
        f"{source}:{geometry}_a{strength:g}"
        for source in SOURCES
        for geometry in GEOMETRIES
        for strength in STRENGTHS
    ]


def run_report(
    config: NonlinearLocalConfig,
    seeds: list[int],
    horizons: list[int],
    candidate_counts: list[int],
    *,
    tasks: int,
    calibration_samples: int,
) -> dict[str, Any]:
    metric_names = _metric_names()
    records: list[dict[str, Any]] = []
    matrix_diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        none_model, train = fitted["none"]
        calibration_generator = torch.Generator().manual_seed(970_000 + int(seed))
        calibration = _sample_split(calibration_samples, calibration_generator, config.noise_std)
        matrices: dict[str, torch.Tensor] = {"euclidean": torch.eye(config.state_dim)}
        matrix_diagnostics[str(seed)] = {}
        for source_index, (source, split) in enumerate((("in_sample", train), ("heldout_calibration", calibration))):
            precision, diagnostics = _residual_precision(none_model, split, 980_000 + int(seed) * 10 + source_index)
            random_precision = diagnostics.pop("random_precision")
            source_matrices = _strength_matrices(precision, random_precision)
            matrix_diagnostics[str(seed)][source] = diagnostics
            for strength in STRENGTHS:
                aligned = source_matrices[f"aligned_a{strength:g}"]
                randomized = source_matrices[f"random_basis_a{strength:g}"]
                matrices[f"{source}:aligned_a{strength:g}"] = aligned
                matrices[f"{source}:random_basis_a{strength:g}"] = randomized
                eigenvalues_a = torch.linalg.eigvalsh((aligned + aligned.T) / 2)
                eigenvalues_r = torch.linalg.eigvalsh((randomized + randomized.T) / 2)
                matrix_diagnostics[str(seed)][f"{source}:a{strength:g}"] = {
                    "same_spectrum_max_abs_error": float((eigenvalues_a - eigenvalues_r).abs().max()),
                    "relative_basis_difference": float((aligned - randomized).norm() / aligned.norm().clamp_min(1e-12)),
                    "aligned_condition_number": float(eigenvalues_a.max() / eigenvalues_a.clamp_min(1e-12).min()),
                    "random_condition_number": float(eigenvalues_r.max() / eigenvalues_r.clamp_min(1e-12).min()),
                }

        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                for count_index, candidates in enumerate(candidate_counts):
                    generator = torch.Generator().manual_seed(
                        990_000 + int(seed) * 100_000 + shift_index * 10_000 + int(horizon) * 100 + count_index
                    )
                    batch = _make_task_batch(tasks, int(candidates), int(horizon), generator, state_shift)
                    for mode, (model, _train) in fitted.items():
                        predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                        for metric in metric_names:
                            planning = _evaluate_metric(
                                predicted,
                                batch["true_endpoints"],
                                batch["target_endpoint"],
                                "residual_strength",
                                matrices[metric],
                            )
                            source = "none" if metric == "euclidean" else metric.split(":", 1)[0]
                            geometry = "euclidean" if metric == "euclidean" else metric.split(":", 1)[1].rsplit("_a", 1)[0]
                            strength = 0.0 if metric == "euclidean" else float(metric.rsplit("_a", 1)[1])
                            records.append({
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": metric,
                                "source": source,
                                "geometry": geometry,
                                "strength": strength,
                                "candidate_count": int(candidates),
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": planning,
                            })

    record_map = {
        (r["model_mode"], r["metric"], r["candidate_count"], float(r["state_shift"]), r["planning_horizon"], r["seed"]): r
        for r in records
    }
    summary: dict[str, Any] = {}
    for mode in MODEL_CONDITIONS:
        summary[mode] = {}
        for metric in metric_names:
            summary[mode][metric] = {}
            for candidates in candidate_counts:
                summary[mode][metric][str(candidates)] = {}
                for shift in SHIFTS:
                    summary[mode][metric][str(candidates)][str(shift)] = {}
                    for horizon in horizons:
                        rows = [record_map[(mode, metric, int(candidates), float(shift), int(horizon), seed)] for seed in seeds]
                        regrets = np.asarray([row["planning"]["planning_regret"] for row in rows], dtype=np.float64)
                        summary[mode][metric][str(candidates)][str(shift)][str(horizon)] = {
                            "n_seeds": int(len(rows)),
                            "planning_regret_mean": float(regrets.mean()),
                            "planning_regret_std": float(regrets.std(ddof=1)),
                            "goal_success_rate_mean": float(np.mean([row["planning"]["goal_success_rate"] for row in rows])),
                            "oracle_choice_agreement_mean": float(np.mean([row["planning"]["oracle_choice_agreement"] for row in rows])),
                        }

    basis: dict[str, Any] = {}
    basis_p: dict[str, float] = {}
    versus_euclidean: dict[str, Any] = {}
    versus_euclidean_p: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for source in SOURCES:
            for strength in STRENGTHS:
                aligned = f"{source}:aligned_a{strength:g}"
                random = f"{source}:random_basis_a{strength:g}"
                for candidates in candidate_counts:
                    for shift in SHIFTS:
                        for horizon in horizons:
                            values = np.asarray([
                                record_map[(mode, aligned, candidates, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                - record_map[(mode, random, candidates, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                for seed in seeds
                            ], dtype=np.float64)
                            key = f"{mode}:{source}:a={strength:g}:c={candidates}:shift={shift:g}:h={horizon}"
                            basis[key] = {"model_mode": mode, "source": source, "strength": float(strength), "candidate_count": int(candidates), "state_shift": float(shift), "planning_horizon": int(horizon), **_paired_summary(values, 991_000 + candidates + horizon, "aligned_minus_random")}
                            basis_p[key] = basis[key]["exact_two_sided_sign_flip_p"]
                            for metric in (aligned, random):
                                euclidean_values = np.asarray([
                                    record_map[(mode, metric, candidates, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                    - record_map[(mode, "euclidean", candidates, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                    for seed in seeds
                                ], dtype=np.float64)
                                euclidean_key = f"{mode}:{metric}:c={candidates}:shift={shift:g}:h={horizon}"
                                versus_euclidean[euclidean_key] = {"model_mode": mode, "metric": metric, "candidate_count": int(candidates), "state_shift": float(shift), "planning_horizon": int(horizon), **_paired_summary(euclidean_values, 992_000 + candidates + horizon, "metric_minus_euclidean")}
                                versus_euclidean_p[euclidean_key] = versus_euclidean[euclidean_key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(basis_p).items():
        basis[key]["holm_p_global"] = value
    for key, value in _holm(versus_euclidean_p).items():
        versus_euclidean[key]["holm_p_global"] = value

    focused_keys = [
        f"none:heldout_calibration:a=1:c={candidates}:shift={max(SHIFTS):g}:h={max(horizons)}"
        for candidates in candidate_counts
    ]
    focused = {key: basis[key] for key in focused_keys}
    for key, value in _holm({key: item["exact_two_sided_sign_flip_p"] for key, item in focused.items()}).items():
        focused[key]["holm_p_focused_family"] = value

    report = {
        "artifact": "nonlinear_residual_candidate_sensitivity",
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "model_conditions": list(MODEL_CONDITIONS),
            "sources": list(SOURCES),
            "geometries": list(GEOMETRIES),
            "strengths": [float(value) for value in STRENGTHS],
            "candidate_counts": [int(value) for value in candidate_counts],
            "state_shifts": [float(value) for value in SHIFTS],
            "horizons": [int(value) for value in horizons],
            "planning_tasks_per_cell": int(tasks),
            "calibration_samples": int(calibration_samples),
            "matched_spectrum_trace_condition_number": True,
            "shared_candidates_across_metrics": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "matrix_diagnostics": matrix_diagnostics,
        "records": records,
        "summary": summary,
        "paired_aligned_minus_random": basis,
        "focused_ood_aligned_minus_random": focused,
        "paired_metrics_minus_euclidean": versus_euclidean,
        "claim_boundary": [
            "This is a nonlinear synthetic candidate-ranking diagnostic, not online LeWM or PushT control.",
            "Aligned and random metrics match trace, spectrum, and condition number; only the eigenbasis is randomized.",
            "Candidate-count cells use independent deterministic task batches, so cross-count changes are robustness comparisons rather than paired treatment effects.",
            "The focused OOD family is narrow and exploratory; the global correction across all cells is retained.",
        ],
    }
    primary_key = focused_keys[-1]
    primary = focused[primary_key]
    report["interpretation"] = [
        f"At the largest candidate set (c={candidate_counts[-1]}), heldout OOD h={max(horizons)} strength=1 aligned-minus-random regret is {primary['mean_delta_aligned_minus_random']:.6g}, with {primary['n_negative_deltas']}/{primary['n_paired_seeds']} negative seed deltas and focused-family Holm p={primary['holm_p_focused_family']:.6g}.",
        "The candidate-count sweep is interpreted as a robustness profile: a sign-consistent aligned-minus-random contrast across counts is stronger evidence than a single candidate-set result, while a count-dependent sign reversal limits the claim to a search-density interaction.",
    ]
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear residual-aware candidate-count sensitivity",
        "",
        "This focused audit varies candidate-set size while retaining heldout residual calibration and matched-spectrum random-basis controls.",
        "",
        "## Regret summary",
        "",
        "| model | metric | candidates | shift | h | n | regret mean +/- sd |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for source in SOURCES:
            for geometry in GEOMETRIES:
                for strength in STRENGTHS:
                    metric = f"{source}:{geometry}_a{strength:g}"
                    for candidates in report["protocol"]["candidate_counts"]:
                        for shift in SHIFTS:
                            for horizon in report["protocol"]["horizons"]:
                                value = report["summary"][mode][metric][str(candidates)][str(shift)][str(horizon)]
                                lines.append(f"| {mode} | {metric} | {candidates} | {shift:g} | {horizon} | {value['n_seeds']} | {value['planning_regret_mean']:.6g} +/- {value['planning_regret_std']:.6g} |")
    lines.extend([
        "", "## Focused OOD basis comparison", "",
        "Negative aligned-minus-random regret favors the residual-aligned basis. The focused Holm correction is within the three candidate-count cells only.", "",
        "| candidates | mean delta | negative fraction | sign-flip p | focused Holm p | global Holm p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for value in report["focused_ood_aligned_minus_random"].values():
        lines.append(f"| {value['candidate_count']} | {value['mean_delta_aligned_minus_random']:.6g} | {value['negative_delta_fraction']:.6g} | {value['exact_two_sided_sign_flip_p']:.6g} | {value['holm_p_focused_family']:.6g} | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["interpretation"]], "", "## Claim boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=list(CANDIDATE_COUNTS))
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--local-anchors", type=int, default=128)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--predictor-weight", type=float, default=0.1)
    args = parser.parse_args()
    config = NonlinearLocalConfig(epochs=args.epochs, train_samples=args.train_samples, validation_samples=args.validation_samples, test_samples=args.test_samples, local_anchors=args.local_anchors, alignment_weight=args.alignment_weight, predictor_weight=args.predictor_weight)
    print("Running residual-aware candidate-count sensitivity: heldout calibration and matched-spectrum random basis.", flush=True)
    report = run_report(config, [int(seed) for seed in args.seeds], [int(horizon) for horizon in args.horizons], [int(count) for count in args.candidate_counts], tasks=args.tasks, calibration_samples=args.calibration_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
