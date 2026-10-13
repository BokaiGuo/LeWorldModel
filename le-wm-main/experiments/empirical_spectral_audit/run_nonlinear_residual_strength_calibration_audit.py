"""Mechanism audit for residual-aware planner metrics.

This follow-up separates three explanations for the residual-precision signal:

1. residual *strength* (mixing the residual metric with Euclidean);
2. residual eigenbasis alignment (same spectrum, independently randomized
   eigenbasis control);
3. calibration leakage (in-sample training residuals versus an independent
   calibration split from the same data-generating process).

Every metric is evaluated on shared candidate sets and shared targets.  The
experiment is a CPU synthetic candidate-ranking diagnostic, not online LeWM,
PushT control, or a causal nuisance intervention.
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
    from experiments.empirical_spectral_audit.run_nonlinear_residual_aware_planner_ablation import (
        MODEL_CONDITIONS,
        SHIFTS,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model, _sample_split
    from run_nonlinear_residual_aware_planner_ablation import (
        MODEL_CONDITIONS,
        SHIFTS,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )


STRENGTHS = (0.25, 0.50, 0.75, 1.00)
SOURCES = ("in_sample", "heldout_calibration")
GEOMETRIES = ("aligned", "random_basis")


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


def _normalize_trace(matrix: torch.Tensor) -> torch.Tensor:
    dimension = matrix.size(0)
    return matrix * (float(dimension) / matrix.trace().clamp_min(1e-8))


def _residual_precision(
    model: Any,
    split: dict[str, torch.Tensor],
    random_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        residual = split["target"].float() - model(split["state"], split["action"]).float()
    centered = residual - residual.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(residual.size(0) - 1, 1)
    dimension = int(residual.size(1))
    ridge = 1e-3 * covariance.trace().clamp_min(1e-8) / float(dimension)
    precision = _normalize_trace(torch.linalg.inv(covariance + ridge * torch.eye(dimension)))
    eigenvalues, eigenvectors = torch.linalg.eigh(precision)
    generator = torch.Generator().manual_seed(int(random_seed))
    random_basis, _ = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator))
    random_precision = _normalize_trace(random_basis @ torch.diag(eigenvalues) @ random_basis.T)
    diagnostics = {
        "residual_count": int(residual.size(0)),
        "residual_trace": float(covariance.trace()),
        "ridge": float(ridge),
        "precision_eigenvalues": [float(value) for value in eigenvalues],
        "random_basis_seed": int(random_seed),
        "matrix_condition_number": float(eigenvalues.max() / eigenvalues.clamp_min(1e-12).min()),
        "metric_estimation": "targets_minus_frozen_model_prediction",
    }
    # The eigenvectors are returned only for diagnostics and are not used to
    # construct the aligned matrix: this makes the random-basis control exact.
    diagnostics["precision_eigenvector_trace"] = float(eigenvectors.square().sum())
    return precision, {**diagnostics, "random_precision": random_precision}


def _strength_matrices(
    precision: torch.Tensor,
    random_precision: torch.Tensor,
) -> dict[str, torch.Tensor]:
    identity = torch.eye(precision.size(0), dtype=precision.dtype, device=precision.device)
    matrices: dict[str, torch.Tensor] = {"euclidean": identity}
    for strength in STRENGTHS:
        suffix = f"{strength:g}"
        matrices[f"aligned_a{suffix}"] = _normalize_trace((1.0 - strength) * identity + strength * precision)
        matrices[f"random_basis_a{suffix}"] = _normalize_trace((1.0 - strength) * identity + strength * random_precision)
    return matrices


def _matrix_diagnostics(
    aligned: torch.Tensor,
    randomized: torch.Tensor,
) -> dict[str, float]:
    aligned_eigenvalues = torch.linalg.eigvalsh((aligned + aligned.T) / 2)
    random_eigenvalues = torch.linalg.eigvalsh((randomized + randomized.T) / 2)
    aligned_condition = aligned_eigenvalues.max() / aligned_eigenvalues.clamp_min(1e-12).min()
    random_condition = random_eigenvalues.max() / random_eigenvalues.clamp_min(1e-12).min()
    return {
        "aligned_trace": float(aligned.trace()),
        "random_trace": float(randomized.trace()),
        "same_spectrum_max_abs_error": float((aligned_eigenvalues - random_eigenvalues).abs().max()),
        "aligned_condition_number": float(aligned_condition),
        "random_condition_number": float(random_condition),
        "relative_basis_difference": float((aligned - randomized).norm() / aligned.norm().clamp_min(1e-12)),
    }


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


def run_report(
    config: NonlinearLocalConfig,
    seeds: list[int],
    horizons: list[int],
    *,
    tasks: int,
    candidates: int,
    calibration_samples: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    matrix_diagnostics: dict[str, Any] = {}
    started = time.time()
    metric_names = ["euclidean"] + [f"{source}:{geometry}_a{strength:g}" for source in SOURCES for geometry in GEOMETRIES for strength in STRENGTHS]
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _best_epoch, _best_validation, _random_scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        none_model, train = fitted["none"]
        calibration_generator = torch.Generator().manual_seed(910_000 + int(seed))
        calibration = _sample_split(calibration_samples, calibration_generator, config.noise_std)
        source_splits = {"in_sample": train, "heldout_calibration": calibration}
        matrices: dict[str, torch.Tensor] = {"euclidean": torch.eye(config.state_dim)}
        matrix_diagnostics[str(seed)] = {}
        for source_index, source in enumerate(SOURCES):
            precision, diagnostics = _residual_precision(none_model, source_splits[source], 920_000 + int(seed) * 10 + source_index)
            random_precision = diagnostics.pop("random_precision")
            source_matrices = _strength_matrices(precision, random_precision)
            for strength in STRENGTHS:
                aligned_name = f"aligned_a{strength:g}"
                random_name = f"random_basis_a{strength:g}"
                matrices[f"{source}:{aligned_name}"] = source_matrices[aligned_name]
                matrices[f"{source}:{random_name}"] = source_matrices[random_name]
                matrix_diagnostics[str(seed)][f"{source}:a{strength:g}"] = _matrix_diagnostics(
                    source_matrices[aligned_name], source_matrices[random_name]
                )
            matrix_diagnostics[str(seed)][source] = diagnostics

        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(930_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon))
                batch = _make_task_batch(tasks, candidates, horizon, generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    for metric in metric_names:
                        if metric == "euclidean":
                            matrix = matrices[metric]
                            source = "none"
                            geometry = "euclidean"
                            strength = 0.0
                        else:
                            source, geometry_strength = metric.split(":", 1) if ":" in metric else (None, metric)
                            # metric_names are source-qualified below; this branch
                            # keeps the serialized record human-readable.
                            if source is None:
                                raise RuntimeError(f"unqualified metric: {metric}")
                            geometry, strength_text = geometry_strength.rsplit("_a", 1)
                            strength = float(strength_text)
                            matrix = matrices[metric]
                        planning = _evaluate_metric(predicted, batch["true_endpoints"], batch["target_endpoint"], "residual_strength", matrix)
                        records.append({
                            "seed": int(seed),
                            "model_mode": mode,
                            "metric": metric,
                            "source": source,
                            "geometry": geometry,
                            "strength": float(strength),
                            "state_shift": float(state_shift),
                            "planning_horizon": int(horizon),
                            "planning": planning,
                        })

    # Build summaries from records.  ``euclidean`` is shared by both sources.
    record_map = {
        (r["model_mode"], r["metric"], float(r["state_shift"]), int(r["planning_horizon"]), int(r["seed"])): r
        for r in records
    }
    summary: dict[str, Any] = {}
    for mode in MODEL_CONDITIONS:
        summary[mode] = {}
        for metric in metric_names:
            summary[mode][metric] = {}
            for shift in SHIFTS:
                summary[mode][metric][str(shift)] = {}
                for horizon in horizons:
                    rows = [record_map[(mode, metric, float(shift), int(horizon), seed)] for seed in seeds]
                    regrets = np.asarray([row["planning"]["planning_regret"] for row in rows], dtype=np.float64)
                    summary[mode][metric][str(shift)][str(horizon)] = {
                        "n_seeds": int(len(rows)),
                        "planning_regret_mean": float(regrets.mean()),
                        "planning_regret_std": float(regrets.std(ddof=1)),
                        "goal_success_rate_mean": float(np.mean([row["planning"]["goal_success_rate"] for row in rows])),
                        "oracle_choice_agreement_mean": float(np.mean([row["planning"]["oracle_choice_agreement"] for row in rows])),
                    }

    # Primary family: aligned minus random at the same source/strength.  This
    # is the direct test of eigenbasis alignment after matching spectrum,
    # trace, and condition number.
    basis_contrasts: dict[str, Any] = {}
    basis_pvalues: dict[str, float] = {}
    calibration_contrasts: dict[str, Any] = {}
    calibration_pvalues: dict[str, float] = {}
    euclidean_contrasts: dict[str, Any] = {}
    euclidean_pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for source in SOURCES:
            for strength in STRENGTHS:
                aligned = f"{source}:aligned_a{strength:g}"
                random = f"{source}:random_basis_a{strength:g}"
                for shift in SHIFTS:
                    for horizon in horizons:
                        deltas = np.asarray([
                            record_map[(mode, aligned, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                            - record_map[(mode, random, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ], dtype=np.float64)
                        key = f"{mode}:{source}:a={strength:g}:shift={shift:g}:h={horizon}"
                        basis_contrasts[key] = {"model_mode": mode, "source": source, "strength": float(strength), "state_shift": float(shift), "planning_horizon": int(horizon), **_paired_summary(deltas, 940_000 + horizon, "aligned_minus_random")}
                        basis_pvalues[key] = basis_contrasts[key]["exact_two_sided_sign_flip_p"]
                        in_metric = f"in_sample:aligned_a{strength:g}"
                        heldout_metric = f"heldout_calibration:aligned_a{strength:g}"
                        calibration_delta = np.asarray([
                            record_map[(mode, heldout_metric, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                            - record_map[(mode, in_metric, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ], dtype=np.float64)
                        calibration_key = f"{mode}:a={strength:g}:shift={shift:g}:h={horizon}"
                        calibration_contrasts[calibration_key] = {"model_mode": mode, "strength": float(strength), "state_shift": float(shift), "planning_horizon": int(horizon), **_paired_summary(calibration_delta, 950_000 + horizon, "heldout_minus_in_sample")}
                        calibration_pvalues[calibration_key] = calibration_contrasts[calibration_key]["exact_two_sided_sign_flip_p"]
                # Strength is compared against Euclidean on every cell; this
                # reveals a monotone or non-monotone strength response.
                for shift in SHIFTS:
                    for horizon in horizons:
                        for metric in (f"in_sample:aligned_a{strength:g}", f"in_sample:random_basis_a{strength:g}", f"heldout_calibration:aligned_a{strength:g}", f"heldout_calibration:random_basis_a{strength:g}"):
                            deltas = np.asarray([
                                record_map[(mode, metric, float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                - record_map[(mode, "euclidean", float(shift), int(horizon), seed)]["planning"]["planning_regret"]
                                for seed in seeds
                            ], dtype=np.float64)
                            key = f"{mode}:{metric}:shift={shift:g}:h={horizon}"
                            euclidean_contrasts[key] = {"model_mode": mode, "metric": metric, "state_shift": float(shift), "planning_horizon": int(horizon), **_paired_summary(deltas, 960_000 + horizon, "metric_minus_euclidean")}
                            euclidean_pvalues[key] = euclidean_contrasts[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(basis_pvalues).items():
        basis_contrasts[key]["holm_p_global"] = value
    for key, value in _holm(calibration_pvalues).items():
        calibration_contrasts[key]["holm_p_global"] = value
    for key, value in _holm(euclidean_pvalues).items():
        euclidean_contrasts[key]["holm_p_global"] = value

    # A narrow, transparent exploratory family is reported separately from
    # the global correction.  It fixes the none model, heldout calibration,
    # OOD shift, and longest horizon, and asks only whether the aligned basis
    # beats its matched-spectrum random basis across the four strengths.
    focused_keys = [
        f"none:heldout_calibration:a={strength:g}:shift={max(SHIFTS):g}:h={max(horizons)}"
        for strength in STRENGTHS
    ]
    focused_pvalues = {key: basis_contrasts[key]["exact_two_sided_sign_flip_p"] for key in focused_keys if key in basis_contrasts}
    focused_basis = {key: basis_contrasts[key] for key in focused_pvalues}
    for key, value in _holm(focused_pvalues).items():
        focused_basis[key]["holm_p_focused_family"] = value

    primary_key = f"none:heldout_calibration:a=1:shift={max(SHIFTS):g}:h={max(horizons)}"
    if primary_key not in basis_contrasts:
        primary_key = next(iter(basis_contrasts))
    primary = basis_contrasts[primary_key]
    focused_primary = focused_basis.get(primary_key, primary)
    report = {
        "artifact": "nonlinear_residual_strength_calibration_audit",
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "model_conditions": list(MODEL_CONDITIONS),
            "sources": list(SOURCES),
            "geometries": list(GEOMETRIES),
            "strengths": [float(value) for value in STRENGTHS],
            "state_shifts": [float(value) for value in SHIFTS],
            "horizons": [int(value) for value in horizons],
            "planning_tasks_per_cell": int(tasks),
            "candidate_sequences_per_task": int(candidates),
            "calibration_samples": int(calibration_samples),
            "shared_candidates_across_all_metrics": True,
            "matched_spectrum_trace_condition_number": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "matrix_diagnostics": matrix_diagnostics,
        "records": records,
        "summary": summary,
        "paired_aligned_minus_random": basis_contrasts,
        "focused_ood_aligned_minus_random": focused_basis,
        "paired_heldout_minus_in_sample": calibration_contrasts,
        "paired_metrics_minus_euclidean": euclidean_contrasts,
        "claim_boundary": [
            "This is a nonlinear synthetic candidate-ranking diagnostic, not online LeWM or PushT control.",
            "Aligned and random-basis metrics have matched trace, eigenvalue spectrum, and condition number; the random control destroys only eigenbasis alignment.",
            "Heldout calibration uses an independent transition split and is not used to train the predictor or score candidate endpoints.",
            "The focused OOD family is a narrow exploratory summary with within-family correction; the global correction remains the appropriate safeguard against searching across all cells.",
            "A non-significant aligned-minus-random contrast does not prove that residual geometry is useless; it fails to identify eigenbasis alignment as a robust source of planner benefit.",
        ],
    }
    report["interpretation"] = [
        f"In the focused none/heldout/OOD-long-horizon family, strength-1 aligned-minus-random regret is {primary['mean_delta_aligned_minus_random']:.6g}, with p={primary['exact_two_sided_sign_flip_p']:.6g} and family-adjusted p={focused_primary.get('holm_p_focused_family', float('nan')):.6g}; the global correction across all basis cells is p={primary['holm_p_global']:.6g}.",
        "The strength sweep is interpreted jointly with the matched-spectrum random control: improvements shared by aligned and random metrics are spectral/conditioning-compatible, whereas an aligned-only improvement is evidence for basis alignment.",
        "Comparing in-sample with independent calibration residuals tests whether any metric ranking survives removal of residual-estimation reuse; it is a robustness check, not a causal intervention.",
    ]
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear residual strength and calibration audit",
        "",
        "This audit varies residual-metric strength, matches the eigenvalue spectrum/trace/condition number of an independently randomized basis control, and compares in-sample with heldout residual calibration. All rows use shared candidate sets and evaluation targets.",
        "",
        "## Primary matrix diagnostics",
        "",
        "| source | strength | trace error | condition(aligned) | condition(random) | max spectrum error | relative basis difference |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, values in report["matrix_diagnostics"].items():
        for key, value in values.items():
            if ":a" not in key:
                continue
            source, strength = key.split(":a", 1)
            lines.append(f"| seed={seed} {source} | {strength} | {abs(value['aligned_trace'] - value['random_trace']):.3g} | {value['aligned_condition_number']:.6g} | {value['random_condition_number']:.6g} | {value['same_spectrum_max_abs_error']:.3g} | {value['relative_basis_difference']:.6g} |")
    lines.extend(["", "## Regret strength sweep", "", "| model | metric | shift | h | n | regret mean +/- sd |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for mode in MODEL_CONDITIONS:
        for source in SOURCES:
            for strength in STRENGTHS:
                for geometry in GEOMETRIES:
                    metric = f"{source}:{geometry}_a{strength:g}"
                    for shift in SHIFTS:
                        for horizon in report["protocol"]["horizons"]:
                            value = report["summary"][mode][metric][str(shift)][str(horizon)]
                            lines.append(f"| {mode} | {metric} | {shift:g} | {horizon} | {value['n_seeds']} | {value['planning_regret_mean']:.6g} +/- {value['planning_regret_std']:.6g} |")
    lines.extend(["", "## Aligned versus matched-spectrum random basis", "", "Negative aligned-minus-random regret favors aligned residual geometry.", "", "| model | source | strength | shift | h | mean delta | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
    for value in report["paired_aligned_minus_random"].values():
        ci = value["bootstrap_ci95"]
        lines.append(f"| {value['model_mode']} | {value['source']} | {value['strength']:g} | {value['state_shift']:g} | {value['planning_horizon']} | {value['mean_delta_aligned_minus_random']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['exact_two_sided_sign_flip_p']:.6g} | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Heldout calibration versus in-sample calibration", "", "Negative heldout-minus-in-sample regret favors the independent calibration estimate.", "", "| model | strength | shift | h | mean delta | 95% bootstrap CI | Holm p |", "| --- | ---: | ---: | ---: | ---: | --- | ---: |"])
    for value in report["paired_heldout_minus_in_sample"].values():
        ci = value["bootstrap_ci95"]
        lines.append(f"| {value['model_mode']} | {value['strength']:g} | {value['state_shift']:g} | {value['planning_horizon']} | {value['mean_delta_heldout_minus_in_sample']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Focused OOD basis family", "", "This is a narrow exploratory family (none model, heldout calibration, OOD shift, longest horizon); its Holm correction is not a substitute for the global correction.", "", "| strength | mean aligned-minus-random | negative fraction | sign-flip p | focused Holm p | global Holm p |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    for value in report["focused_ood_aligned_minus_random"].values():
        lines.append(f"| {value['strength']:g} | {value['mean_delta_aligned_minus_random']:.6g} | {value['negative_delta_fraction']:.6g} | {value['exact_two_sided_sign_flip_p']:.6g} | {value['holm_p_focused_family']:.6g} | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["interpretation"]], "", "## Claim boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=48)
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
    print("Running residual strength/calibration audit: matched-spectrum basis controls and independent residual calibration.", flush=True)
    report = run_report(config, [int(seed) for seed in args.seeds], [int(horizon) for horizon in args.horizons], tasks=args.tasks, candidates=args.candidates, calibration_samples=args.calibration_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
