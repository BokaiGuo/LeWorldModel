"""State-conditioned controllability metric ablation.

The global Gramian audit uses one training-fitted A/B surrogate for all
states.  This experiment instead estimates a local secant Jacobian and local
binary-action effect at each current state, then constructs a per-task
finite-horizon controllability precision metric.  The metric uses only the
predictor and the current state; target endpoints and true future transitions
are withheld until scoring.

This is still a synthetic candidate-set planning proxy, not online LeWM or
PushT control evidence.
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
    from experiments.empirical_spectral_audit.run_nonlinear_controllability_metric_ablation import (
        _controllability_precision,
        _fit_linear_transition,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        _fit_model,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
        _build_metric_matrices,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_controllability_metric_ablation import _controllability_precision, _fit_linear_transition
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import _build_metric_matrices, _make_task_batch, _predicted_endpoints


MODEL_CONDITIONS = ("none", "global", "conditional", "predictor", "both")
METRICS = (
    "euclidean",
    "covariance_precision",
    "global_controllability",
    "local_controllability",
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


@torch.no_grad()
def _local_linearization(
    model: Any,
    state: torch.Tensor,
    eps: float = 2e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-state A and B from model queries at both actions."""

    action_zero = torch.zeros(state.size(0), dtype=torch.long)
    action_one = torch.ones(state.size(0), dtype=torch.long)
    output_zero = model(state, action_zero)
    output_one = model(state, action_one)
    B = (output_one - output_zero).unsqueeze(-1)
    jacobians: list[torch.Tensor] = []
    for action in (action_zero, action_one):
        base = model(state, action)
        columns: list[torch.Tensor] = []
        for coordinate in range(state.size(-1)):
            perturbation = torch.zeros_like(state)
            perturbation[:, coordinate] = float(eps)
            columns.append((model(state + perturbation, action) - base) / float(eps))
        jacobians.append(torch.stack(columns, dim=-1))
    A = (jacobians[0] + jacobians[1]) / 2
    return A, B


@torch.no_grad()
def _local_precision(
    model: Any,
    state: torch.Tensor,
    horizon: int,
    eps: float = 2e-3,
    ridge_fraction: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Construct one PSD precision matrix per current state."""

    A, B = _local_linearization(model, state, eps=eps)
    count, dimension = state.size(0), state.size(1)
    gramian = torch.zeros(count, dimension, dimension, dtype=state.dtype)
    propagated = B @ B.transpose(-1, -2)
    for _ in range(int(horizon)):
        gramian = gramian + propagated
        propagated = A @ propagated @ A.transpose(-1, -2)
    gramian = (gramian + gramian.transpose(-1, -2)) / 2
    ridge = ridge_fraction * gramian.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8) / float(dimension)
    identity = torch.eye(dimension, dtype=state.dtype).expand(count, -1, -1)
    precision = torch.linalg.inv(gramian + ridge[:, None, None] * identity)
    precision = (precision + precision.transpose(-1, -2)) / 2
    precision = precision * (float(dimension) / precision.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8))[:, None, None]
    eigenvalues = torch.linalg.eigvalsh(gramian)
    return precision, {
        "mean_gramian_trace": float(gramian.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
        "mean_gramian_min_eigenvalue": float(eigenvalues[:, 0].mean()),
        "mean_gramian_max_eigenvalue": float(eigenvalues[:, -1].mean()),
        "mean_effective_rank": float(
            (eigenvalues / eigenvalues[:, -1:].clamp_min(1e-12) > 1e-3).float().sum(-1).mean()
        ),
        "ridge_fraction": float(ridge_fraction),
        "secant_eps": float(eps),
    }


@torch.no_grad()
def _evaluate_metric_batch(
    predicted_endpoints: torch.Tensor,
    true_endpoints: torch.Tensor,
    target_endpoint: torch.Tensor,
    metric: str,
    matrix: torch.Tensor | None,
) -> dict[str, float]:
    target = target_endpoint[:, None, :]
    true_difference = true_endpoints - target
    true_squared_euclidean = true_difference.square().sum(dim=-1)
    oracle_index = true_squared_euclidean.argmin(dim=1)
    oracle_distance = true_squared_euclidean.gather(1, oracle_index[:, None]).sqrt().squeeze(1)
    chosen_metric = metric
    if metric == "oracle":
        chosen = oracle_index
    else:
        if matrix is None:
            raise ValueError(f"missing matrix for {metric}")
        predicted_difference = predicted_endpoints - target
        if matrix.dim() == 2:
            costs = torch.einsum("ncd,df,ncf->nc", predicted_difference, matrix, predicted_difference)
        else:
            costs = torch.einsum("ncd,ndf,ncf->nc", predicted_difference, matrix, predicted_difference)
        chosen = costs.argmin(dim=1)
    selected_distance = true_squared_euclidean.gather(1, chosen[:, None]).sqrt().squeeze(1)
    return {
        "selected_goal_distance": float(selected_distance.mean()),
        "oracle_goal_distance": float(oracle_distance.mean()),
        "planning_regret": float((selected_distance - oracle_distance).mean()),
        "goal_success_rate": float((selected_distance <= 0.25).float().mean()),
        "oracle_success_rate": float((oracle_distance <= 0.25).float().mean()),
        "oracle_choice_agreement": float((chosen == oracle_index).float().mean()),
        "target_candidate_choice_rate": float((chosen == 0).float().mean()),
    }


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
    secant_eps: float,
    ridge_fraction: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    local_diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        base_matrices, base_diagnostics = _build_metric_matrices(fitted["none"][1], 90_000 + int(seed))
        A, B = _fit_linear_transition(fitted["none"][1])
        global_matrices = {
            int(horizon): _controllability_precision(A, B, int(horizon))[0]
            for horizon in horizons
        }
        local_diagnostics[str(seed)] = {"base_metric": base_diagnostics, "by_mode_horizon": {}}
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    160_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon)
                )
                batch = _make_task_batch(tasks, candidates, int(horizon), generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    local_matrix, diagnostic = _local_precision(
                        model,
                        batch["initial"],
                        int(horizon),
                        eps=secant_eps,
                        ridge_fraction=ridge_fraction,
                    )
                    local_diagnostics[str(seed)]["by_mode_horizon"][f"{mode}/shift={state_shift:g}/h={horizon}"] = diagnostic
                    matrices = dict(base_matrices)
                    matrices["global_controllability"] = global_matrices[int(horizon)]
                    matrices["local_controllability"] = local_matrix
                    for metric in METRICS:
                        records.append(
                            {
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": metric,
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": _evaluate_metric_batch(
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
            for shift in SHIFTS:
                summary[mode][metric][str(shift)] = {}
                for horizon in horizons:
                    rows = [
                        row
                        for row in records
                        if row["model_mode"] == mode
                        and row["metric"] == metric
                        and row["state_shift"] == shift
                        and row["planning_horizon"] == horizon
                    ]
                    if len(rows) != len(seeds):
                        raise RuntimeError("incomplete local-controllability matrix")
                    summary[mode][metric][str(shift)][str(horizon)] = {
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
            for shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray(
                        [
                            record_map[(mode, metric, shift, horizon, seed)]["planning"]["planning_regret"]
                            - record_map[(mode, "euclidean", shift, horizon, seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    key = f"{mode}:{metric}:shift={shift:g}:h={horizon}"
                    inference[key] = {
                        "model_mode": mode,
                        "metric": metric,
                        "state_shift": float(shift),
                        "planning_horizon": int(horizon),
                        **_paired_summary(deltas, seed=20261026 + int(horizon)),
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
            "secant_eps": float(secant_eps),
            "local_ridge_fraction": float(ridge_fraction),
            "local_metric_uses_current_state_only": True,
            "shared_candidates_across_models_and_metrics": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "local_diagnostics": local_diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": inference,
        "claim_boundary": [
            "Local metrics use predictor queries at the current state and do not use target endpoints or true future transitions.",
            "The local secant Gramian is a model-based diagnostic for one nonlinear synthetic transition, not a causal control estimate.",
            "This is a candidate-set planning proxy, not online LeWM or PushT control.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear local controllability metric ablation",
        "",
        "Per-task local secant Jacobians and binary-action effects are used to construct a finite-horizon Gramian precision metric. The target and true future states are withheld until scoring.",
        "",
        "| model | metric | shift | h | n | regret mean +/- sd | success | oracle agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    values = report["summary"][mode][metric][str(shift)][str(horizon)]
                    lines.append(
                        f"| {mode} | {metric} | {shift:g} | {horizon} | {values['n_seeds']} | {values['planning_regret_mean']:.6g} +/- {values['planning_regret_std']:.6g} | {values['goal_success_rate_mean']:.6g} | {values['oracle_choice_agreement_mean']:.6g} |"
                    )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--secant-eps", type=float, default=2e-3)
    parser.add_argument("--ridge-fraction", type=float, default=1e-2)
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
    print("Running state-conditioned local controllability ablation with shared candidate sets.")
    report = run_report(
        config,
        [int(seed) for seed in args.seeds],
        [int(horizon) for horizon in args.horizons],
        tasks=args.tasks,
        candidates=args.candidates,
        secant_eps=args.secant_eps,
        ridge_fraction=args.ridge_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
