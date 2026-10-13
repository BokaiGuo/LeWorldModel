"""Residual-aware planner metric ablation for the nonlinear proxy.

The frozen LeWM audit found that high prediction-error directions can be more
aligned with nuisance/residual geometry than with the small planner-facing
dynamics bands.  This experiment turns that observation into a falsifiable
planner metric: downweight directions with large training prediction-residual
variance (``residual_precision``), compare it with the opposite
``residual_covariance`` metric, and use an equal-spectrum random-basis control
to test whether residual eigenvector alignment matters.

All matrices are estimated from the training split of the same synthetic
world.  Candidate sets, model checkpoints, seeds, and evaluation targets are
shared across metrics.  This is a synthetic candidate-ranking diagnostic, not
online LeWM or PushT control.
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
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
        MODEL_CONDITIONS,
        SHIFTS,
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import (
        MODEL_CONDITIONS,
        SHIFTS,
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )


METRICS = (
    "euclidean",
    "covariance_precision",
    "reachability",
    "random_spectral",
    "residual_precision",
    "residual_covariance",
    "residual_random_spectral",
    "oracle",
)


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


def _residual_metrics(
    model: Any,
    train: dict[str, torch.Tensor],
    random_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with torch.no_grad():
        prediction = model(train["state"], train["action"])
        residual = train["target"].float() - prediction.float()
    covariance = residual.T @ residual / max(residual.size(0) - 1, 1)
    state_dim = int(residual.size(1))
    ridge = 1e-3 * covariance.trace().clamp_min(1e-8) / float(state_dim)
    residual_covariance = _normalize_trace(covariance + ridge * torch.eye(state_dim))
    residual_precision = _normalize_trace(torch.linalg.inv(covariance + ridge * torch.eye(state_dim)))
    eigenvalues, _ = torch.linalg.eigh(residual_precision)
    generator = torch.Generator().manual_seed(int(random_seed))
    random_basis, _ = torch.linalg.qr(torch.randn(state_dim, state_dim, generator=generator))
    residual_random = _normalize_trace(random_basis @ torch.diag(eigenvalues) @ random_basis.T)
    matrices = {
        "residual_precision": residual_precision,
        "residual_covariance": residual_covariance,
        "residual_random_spectral": residual_random,
    }
    return matrices, {
        "residual_trace": float(covariance.trace()),
        "residual_variance": [float(value) for value in covariance.diag()],
        "residual_precision_eigenvalues": [float(value) for value in eigenvalues],
        "ridge": float(ridge),
        "random_basis_seed": int(random_seed),
        "metric_estimation": "training_targets_minus_frozen_none_model_prediction",
    }


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / (2 ** int(values.size)))


def _paired_summary(values: np.ndarray, seed: int, name: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(10000)])
    return {
        "n_paired_seeds": int(values.size),
        f"mean_delta_{name}": float(values.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
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
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    matrix_diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _best_epoch, _best_validation, _random_scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        matrices, diagnostics = _build_metric_matrices(fitted["none"][1], 70_000 + int(seed))
        residual_matrices, residual_diagnostics = _residual_metrics(
            fitted["none"][0], fitted["none"][1], 80_000 + int(seed)
        )
        matrices.update(residual_matrices)
        matrix_diagnostics[str(seed)] = {"base": diagnostics, "residual": residual_diagnostics}
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(410_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon))
                batch = _make_task_batch(tasks, candidates, horizon, generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    for metric in METRICS:
                        planning = _evaluate_metric(predicted, batch["true_endpoints"], batch["target_endpoint"], metric, matrices.get(metric))
                        records.append({"seed": int(seed), "model_mode": mode, "metric": metric, "state_shift": float(state_shift), "planning_horizon": int(horizon), "planning": planning})

    record_map = {(row["model_mode"], row["metric"], float(row["state_shift"]), int(row["planning_horizon"]), int(row["seed"])): row for row in records}
    summary: dict[str, Any] = {}
    for mode in MODEL_CONDITIONS:
        summary[mode] = {}
        for metric in METRICS:
            summary[mode][metric] = {}
            for state_shift in SHIFTS:
                summary[mode][metric][str(state_shift)] = {}
                for horizon in horizons:
                    rows = [record_map[(mode, metric, float(state_shift), int(horizon), seed)] for seed in seeds]
                    regrets = np.asarray([row["planning"]["planning_regret"] for row in rows], dtype=np.float64)
                    summary[mode][metric][str(state_shift)][str(horizon)] = {
                        "n_seeds": int(len(rows)),
                        "planning_regret_mean": float(regrets.mean()),
                        "planning_regret_std": float(regrets.std(ddof=1)),
                        "goal_success_rate_mean": float(np.mean([row["planning"]["goal_success_rate"] for row in rows])),
                        "oracle_choice_agreement_mean": float(np.mean([row["planning"]["oracle_choice_agreement"] for row in rows])),
                        "target_candidate_choice_rate_mean": float(np.mean([row["planning"]["target_candidate_choice_rate"] for row in rows])),
                    }

    paired_vs_euclidean: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for metric in METRICS[1:-1]:
            for state_shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray([
                        record_map[(mode, metric, float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                        - record_map[(mode, "euclidean", float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                        for seed in seeds
                    ], dtype=np.float64)
                    key = f"{mode}:{metric}:shift={state_shift:g}:h={horizon}"
                    paired_vs_euclidean[key] = {"model_mode": mode, "metric": metric, "state_shift": float(state_shift), "planning_horizon": int(horizon), **_paired_summary(deltas, 20261102 + int(horizon), "target_minus_euclidean")}
                    pvalues[key] = paired_vs_euclidean[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(pvalues).items():
        paired_vs_euclidean[key]["holm_p_global"] = value

    direct_basis: dict[str, Any] = {}
    direct_pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for comparison in (("residual_precision", "residual_random_spectral"), ("residual_precision", "residual_covariance")):
            first, second = comparison
            for state_shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray([
                        record_map[(mode, first, float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                        - record_map[(mode, second, float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                        for seed in seeds
                    ], dtype=np.float64)
                    key = f"{mode}:{first}_minus_{second}:shift={state_shift:g}:h={horizon}"
                    direct_basis[key] = {"model_mode": mode, "first_metric": first, "second_metric": second, "state_shift": float(state_shift), "planning_horizon": int(horizon), **_paired_summary(deltas, 20261103 + int(horizon), "first_minus_second")}
                    direct_pvalues[key] = direct_basis[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(direct_pvalues).items():
        direct_basis[key]["holm_p_global"] = value

    report = {
        "artifact": "nonlinear_residual_aware_planner_metric_ablation",
        "protocol": {**asdict(config), "seeds": [int(seed) for seed in seeds], "model_conditions": list(MODEL_CONDITIONS), "metrics": list(METRICS), "state_shifts": [float(value) for value in SHIFTS], "horizons": [int(value) for value in horizons], "planning_tasks_per_cell": int(tasks), "candidate_sequences_per_task": int(candidates), "shared_candidates_across_metrics": True, "metric_estimates_use": "training transitions only; residual metrics use frozen none-model training residuals", "oracle_is_privileged": True, "synthetic_only": True, "elapsed_seconds": time.time() - started},
        "matrix_diagnostics": matrix_diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": paired_vs_euclidean,
        "paired_residual_metric_inference": direct_basis,
        "claim_boundary": [
            "This is a nonlinear synthetic candidate-set planning proxy, not online LeWM or PushT control.",
            "Residual-aware metrics are estimated from training targets minus a frozen none-model prediction; test endpoints are used only for scoring.",
            "Residual covariance emphasizes high-error directions while residual precision downweights them; neither is a causal nuisance intervention.",
            "Random residual spectral preserves the residual-precision spectrum while destroying its eigenbasis alignment.",
        ],
    }
    euclidean_h8 = summary["none"]["euclidean"]["1.5"]["8"]["planning_regret_mean"]
    precision_h8 = summary["none"]["residual_precision"]["1.5"]["8"]["planning_regret_mean"]
    covariance_h8 = summary["none"]["residual_covariance"]["1.5"]["8"]["planning_regret_mean"]
    random_h8 = summary["none"]["residual_random_spectral"]["1.5"]["8"]["planning_regret_mean"]
    precision_vs_random_h8 = direct_basis["none:residual_precision_minus_residual_random_spectral:shift=1.5:h=8"]
    precision_vs_covariance_h8 = direct_basis["none:residual_precision_minus_residual_covariance:shift=1.5:h=8"]
    report["interpretation"] = [
        f"For the none-model on OOD h=8, residual precision regret is {precision_h8:.6g} versus {euclidean_h8:.6g} for Euclidean, a delta of {precision_h8 - euclidean_h8:.6g}.",
        f"Residual precision is {precision_h8 - covariance_h8:.6g} better than residual covariance at OOD h=8, but the direct Holm-adjusted p-value is {precision_vs_covariance_h8['holm_p_global']:.6g}.",
        f"The equal-spectrum random residual basis has OOD h=8 regret {random_h8:.6g}; precision-minus-random delta is {precision_vs_random_h8['mean_delta_first_minus_second']:.6g} with Holm-adjusted p={precision_vs_random_h8['holm_p_global']:.6g}.",
        "Residual downweighting therefore gives a local synthetic improvement signal, but the current evidence does not identify residual eigenvector alignment as the cause or establish a universal planner metric.",
    ]
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# Nonlinear residual-aware planner metric ablation", "", "Residual-aware metrics are estimated from training residuals of the frozen none-model. All model conditions and metrics share candidate sets, seeds, and evaluation targets.", "", "| model | metric | shift | h | n | regret mean +/- sd | success | oracle agreement |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    value = report["summary"][mode][metric][str(shift)][str(horizon)]
                    lines.append(f"| {mode} | {metric} | {shift:g} | {horizon} | {value['n_seeds']} | {value['planning_regret_mean']:.6g} +/- {value['planning_regret_std']:.6g} | {value['goal_success_rate_mean']:.6g} | {value['oracle_choice_agreement_mean']:.6g} |")
    lines.extend(["", "## Paired inference versus Euclidean", "", "| model | metric | shift | h | n | mean delta | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |"])
    for value in report["paired_regret_inference_vs_euclidean"].values():
        ci = value["bootstrap_ci95"]
        lines.append(f"| {value['model_mode']} | {value['metric']} | {value['state_shift']:g} | {value['planning_horizon']} | {value['n_paired_seeds']} | {value['mean_delta_target_minus_euclidean']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['exact_two_sided_sign_flip_p']:.6g} | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Direct residual-basis contrasts", "", "Negative first-minus-second deltas favor the first metric for regret.", "", "| model | first | second | shift | h | mean delta | 95% bootstrap CI | Holm p |", "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"])
    for value in report["paired_residual_metric_inference"].values():
        ci = value["bootstrap_ci95"]
        lines.append(f"| {value['model_mode']} | {value['first_metric']} | {value['second_metric']} | {value['state_shift']:g} | {value['planning_horizon']} | {value['mean_delta_first_minus_second']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['holm_p_global']:.6g} |")
    lines.extend(["", "## Claim boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["interpretation"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--local-anchors", type=int, default=128)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--predictor-weight", type=float, default=0.1)
    args = parser.parse_args()
    config = NonlinearLocalConfig(epochs=args.epochs, train_samples=args.train_samples, validation_samples=args.validation_samples, test_samples=args.test_samples, local_anchors=args.local_anchors, alignment_weight=args.alignment_weight, predictor_weight=args.predictor_weight)
    print("Running residual-aware planner metric ablation: Euclidean, covariance, residual precision/covariance, random residual basis, and oracle.", flush=True)
    report = run_report(config, [int(seed) for seed in args.seeds], [int(horizon) for horizon in args.horizons], tasks=args.tasks, candidates=args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
