"""Finite-horizon controllability metric ablation for the nonlinear proxy.

The previous planner audit used a diagonal action-effect weighting.  This
follow-up fits a training-only local linear transition surrogate and builds a
finite-horizon controllability Gramian

    W_H = sum_{i=0}^{H-1} A^i B B^T (A^T)^i.

The planner uses the normalized precision of W_H as a goal metric.  Directions
with low estimated reachability receive more weight because endpoint error in
those directions is harder to repair.  The metric is evaluated on the same
shared reachable-target candidate sets as the primary planner audit.

This remains a nonlinear synthetic candidate-set planning proxy; it is not an
online LeWM/PushT controller and it does not establish a causal mechanism.
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
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import (
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )


MODEL_CONDITIONS = ("none", "global", "conditional", "predictor", "both")
METRICS = (
    "euclidean",
    "covariance_precision",
    "reachability",
    "controllability_precision",
    "random_spectral",
    "oracle",
)
SHIFTS = (0.0, 1.5)


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


def _fit_linear_transition(
    train: dict[str, torch.Tensor], ridge_fraction: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit y ~= intercept + A x + B a from training transitions."""

    state = train["state"].float()
    action = train["action"].reshape(-1, 1).float()
    target = train["target"].float()
    design = torch.cat((state, action, torch.ones_like(action)), dim=1)
    gram = design.T @ design
    ridge = ridge_fraction * gram.trace().clamp_min(1e-8) / float(design.size(1))
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(design.size(1)), design.T @ target
    )
    state_dim = state.size(-1)
    A = coefficients[:state_dim].T
    B = coefficients[state_dim : state_dim + 1].T
    return A, B


def _controllability_precision(
    A: torch.Tensor,
    B: torch.Tensor,
    horizon: int,
    ridge_fraction: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return normalized W_H precision and auditable Gramian diagnostics."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    dimension = A.size(0)
    gramian = torch.zeros(dimension, dimension, dtype=A.dtype)
    propagated = B @ B.T
    for _ in range(horizon):
        gramian = gramian + propagated
        propagated = A @ propagated @ A.T
    gramian = (gramian + gramian.T) / 2
    ridge = ridge_fraction * gramian.trace().clamp_min(1e-8) / float(dimension)
    precision = torch.linalg.inv(gramian + ridge * torch.eye(dimension))
    precision = _normalize_trace((precision + precision.T) / 2)
    eigenvalues = torch.linalg.eigvalsh(gramian)
    precision_eigenvalues = torch.linalg.eigvalsh(precision)
    diagnostics = {
        "horizon": int(horizon),
        "gramian_eigenvalues": [float(value) for value in eigenvalues],
        "precision_eigenvalues": [float(value) for value in precision_eigenvalues],
        "gramian_trace": float(gramian.trace()),
        "regularization": float(ridge),
        "linear_surrogate": "training target ~= intercept + A state + B binary action",
    }
    return precision, diagnostics


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / (2 ** int(values.size)))


def _paired_summary(values: np.ndarray, seed: int, repeats: int = 10000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(repeats)],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(values.size),
        "mean_delta_target_minus_euclidean": float(values.mean()),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
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
    gramian_diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        base_matrices, base_diagnostics = _build_metric_matrices(
            fitted["none"][1], 80_000 + int(seed)
        )
        A, B = _fit_linear_transition(fitted["none"][1])
        controllability_matrices: dict[int, torch.Tensor] = {}
        gramian_diagnostics[str(seed)] = {
            "action_linear_surrogate": {
                "A": A.tolist(),
                "B": B.tolist(),
            },
            "base_metric": base_diagnostics,
            "by_horizon": {},
        }
        for horizon in horizons:
            matrix, diagnostics = _controllability_precision(A, B, int(horizon))
            controllability_matrices[int(horizon)] = matrix
            gramian_diagnostics[str(seed)]["by_horizon"][str(horizon)] = diagnostics

        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    140_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon)
                )
                batch = _make_task_batch(tasks, candidates, int(horizon), generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    matrices = dict(base_matrices)
                    matrices["controllability_precision"] = controllability_matrices[int(horizon)]
                    for metric in METRICS:
                        records.append(
                            {
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": metric,
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": _evaluate_metric(
                                    predicted,
                                    batch["true_endpoints"],
                                    batch["target_endpoint"],
                                    metric,
                                    matrices.get(metric),
                                ),
                            }
                        )

    summary: dict[str, Any] = {}
    for mode in MODEL_CONDITIONS:
        summary[mode] = {}
        for metric in METRICS:
            summary[mode][metric] = {}
            for state_shift in SHIFTS:
                summary[mode][metric][str(state_shift)] = {}
                for horizon in horizons:
                    rows = [
                        row
                        for row in records
                        if row["model_mode"] == mode
                        and row["metric"] == metric
                        and row["state_shift"] == state_shift
                        and row["planning_horizon"] == horizon
                    ]
                    if len(rows) != len(seeds):
                        raise RuntimeError("incomplete controllability metric matrix")
                    summary[mode][metric][str(state_shift)][str(horizon)] = {
                        "n_seeds": len(rows),
                        "planning_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in rows])),
                        "planning_regret_std": float(np.std([r["planning"]["planning_regret"] for r in rows], ddof=1)),
                        "goal_success_rate_mean": float(np.mean([r["planning"]["goal_success_rate"] for r in rows])),
                        "oracle_choice_agreement_mean": float(np.mean([r["planning"]["oracle_choice_agreement"] for r in rows])),
                    }

    record_map = {
        (r["model_mode"], r["metric"], r["state_shift"], r["planning_horizon"], r["seed"]): r
        for r in records
    }
    inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for metric in METRICS[1:-1]:
            for state_shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray(
                        [
                            record_map[(mode, metric, state_shift, horizon, seed)]["planning"]["planning_regret"]
                            - record_map[(mode, "euclidean", state_shift, horizon, seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    key = f"{mode}:{metric}:shift={state_shift:g}:h={horizon}"
                    inference[key] = {
                        "model_mode": mode,
                        "metric": metric,
                        "state_shift": float(state_shift),
                        "planning_horizon": int(horizon),
                        **_paired_summary(deltas, seed=20261024 + int(horizon)),
                    }
                    raw_pvalues[key] = inference[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        inference[key]["holm_p_global"] = value

    report = {
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "model_conditions": list(MODEL_CONDITIONS),
            "metrics": list(METRICS),
            "state_shifts": [float(value) for value in SHIFTS],
            "horizons": [int(value) for value in horizons],
            "planning_tasks_per_cell": int(tasks),
            "candidate_sequences_per_task": int(candidates),
            "candidate_zero_is_reachable_target_sequence": True,
            "shared_candidates_across_models_and_metrics": True,
            "gramian_estimates_use": "training transitions only",
            "oracle_is_privileged": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "gramian_diagnostics": gramian_diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": inference,
        "claim_boundary": [
            "The controllability Gramian is a training-only linear surrogate for one nonlinear synthetic transition.",
            "The precision metric emphasizes estimated low-reachability directions; this is a hypothesis test, not a universal planner prescription.",
            "The oracle uses true transition endpoints and is a privileged reference.",
            "This is a candidate-set planning proxy, not online LeWM or PushT control.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear finite-horizon controllability metric ablation",
        "",
        "A training-only linear transition surrogate is used to construct a finite-horizon controllability-Gramian precision metric. All models and metrics share candidate tasks.",
        "",
        "| model | metric | shift | h | n | regret mean +/- sd | success | oracle agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for state_shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    values = report["summary"][mode][metric][str(state_shift)][str(horizon)]
                    lines.append(
                        f"| {mode} | {metric} | {state_shift:g} | {horizon} | {values['n_seeds']} | {values['planning_regret_mean']:.6g} +/- {values['planning_regret_std']:.6g} | {values['goal_success_rate_mean']:.6g} | {values['oracle_choice_agreement_mean']:.6g} |"
                    )
    lines.extend(
        [
            "",
            "## Paired regret inference versus Euclidean",
            "",
            "The paired unit is the seed; Holm correction covers learned metric contrasts across model conditions, shifts, and horizons.",
            "",
            "| model | metric | shift | h | n | mean delta | 95% bootstrap CI | exact sign-flip p | Holm p |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for values in report["paired_regret_inference_vs_euclidean"].values():
        ci = values["bootstrap_ci95"]
        lines.append(
            f"| {values['model_mode']} | {values['metric']} | {values['state_shift']:g} | {values['planning_horizon']} | {values['n_paired_seeds']} | {values['mean_delta_target_minus_euclidean']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |"
        )
    lines.extend(["", "## Claim boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
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
    config = NonlinearLocalConfig(
        epochs=args.epochs,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        local_anchors=args.local_anchors,
        alignment_weight=args.alignment_weight,
        predictor_weight=args.predictor_weight,
    )
    print("Running finite-horizon controllability-Gramian metric ablation with shared candidate sets.")
    report = run_report(
        config,
        [int(seed) for seed in args.seeds],
        [int(horizon) for horizon in args.horizons],
        tasks=args.tasks,
        candidates=args.candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear controllability metric ablation\n\n"
        "Training-only finite-horizon Gramian precision compared with covariance and action-reachability metrics.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
