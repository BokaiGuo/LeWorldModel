"""Planner-facing metric ablation for the nonlinear synthetic world.

This experiment isolates the candidate-ranking metric from the learned
predictor.  Every model condition is evaluated on exactly the same initial
states, reachable target action sequences, and distractor candidates.  The
metric alternatives are estimated from training transitions only:

* ``euclidean``: unweighted latent endpoint distance;
* ``covariance_precision``: full covariance-whitened endpoint distance;
* ``reachability``: diagonal weighting by action-induced target variation;
* ``random_spectral``: the covariance metric with an independent orthogonal
  eigenbasis, preserving its spectrum but destroying its coordinate alignment;
* ``oracle``: privileged true-transition endpoint distance.

The result is a synthetic candidate-set planning proxy.  It is deliberately
not described as online LeWM/PushT control or as a nuisance-identification
experiment: this world has no nuisance labels, and ``random_spectral`` is the
negative spectral control.
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
        true_transition,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model, true_transition


METRICS = (
    "euclidean",
    "covariance_precision",
    "reachability",
    "random_spectral",
    "oracle",
)
MODEL_CONDITIONS = ("none", "global", "conditional", "predictor", "both")
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


def _covariance(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(values.size(0) - 1, 1)


def _build_metric_matrices(
    train: dict[str, torch.Tensor], random_seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Estimate all non-oracle metrics from training transitions only."""

    targets = train["target"].float()
    state_dim = targets.size(-1)
    covariance = _covariance(targets)
    ridge = 1e-3 * covariance.trace().clamp_min(1e-8) / float(state_dim)
    covariance_precision = torch.linalg.inv(covariance + ridge * torch.eye(state_dim))
    covariance_precision = _normalize_trace(covariance_precision)

    labels = train["action"].reshape(-1).long()
    group_zero = targets[labels == 0]
    group_one = targets[labels == 1]
    if not len(group_zero) or not len(group_one):
        raise ValueError("training split must contain both action groups")
    action_effect = group_one.mean(dim=0) - group_zero.mean(dim=0)
    target_variance = covariance.diag().clamp_min(1e-6)
    reachability_weights = action_effect.square() / target_variance
    reachability_weights = reachability_weights / reachability_weights.mean().clamp_min(1e-8)
    reachability = torch.diag(reachability_weights)

    generator = torch.Generator().manual_seed(int(random_seed))
    random_basis, _ = torch.linalg.qr(torch.randn(state_dim, state_dim, generator=generator))
    eigenvalues, _ = torch.linalg.eigh(covariance_precision)
    random_spectral = random_basis @ torch.diag(eigenvalues) @ random_basis.T
    random_spectral = _normalize_trace(random_spectral)

    matrices = {
        "euclidean": torch.eye(state_dim),
        "covariance_precision": covariance_precision,
        "reachability": reachability,
        "random_spectral": random_spectral,
    }
    for name, matrix in matrices.items():
        if not torch.isfinite(matrix).all():
            raise FloatingPointError(f"non-finite metric matrix: {name}")
        minimum_eigenvalue = float(torch.linalg.eigvalsh((matrix + matrix.T) / 2).min())
        if minimum_eigenvalue < -1e-6:
            raise ValueError(f"metric is not PSD: {name} ({minimum_eigenvalue})")
    diagnostics = {
        "target_variance": [float(value) for value in target_variance],
        "action_effect": [float(value) for value in action_effect],
        "reachability_weights": [float(value) for value in reachability_weights],
        "covariance_precision_eigenvalues": [float(value) for value in eigenvalues],
        "random_basis_seed": int(random_seed),
        "metric_estimation": "training_targets_only",
    }
    return matrices, diagnostics


def _make_task_batch(
    count: int,
    candidates: int,
    horizon: int,
    generator: torch.Generator,
    state_shift: float,
) -> dict[str, torch.Tensor]:
    """Create shared reachable goals and candidate action sequences."""

    if candidates < 2:
        raise ValueError("candidate set must contain at least two sequences")
    initial = torch.randn(count, 4, generator=generator) + float(state_shift)
    target_actions = (torch.rand(count, horizon, generator=generator) < 0.5).long()
    distractor_actions = (
        torch.rand(count, candidates - 1, horizon, generator=generator) < 0.5
    ).long()
    actions = torch.cat((target_actions[:, None, :], distractor_actions), dim=1)
    flat_state = initial[:, None, :].expand(-1, candidates, -1).reshape(-1, 4)
    flat_actions = actions.reshape(-1, horizon)
    states = [flat_state]
    current = flat_state
    for step in range(horizon):
        current = true_transition(current, flat_actions[:, step])
        states.append(current)
    endpoints = states[-1].reshape(count, candidates, -1)
    return {
        "initial": initial,
        "actions": actions,
        "true_endpoints": endpoints,
        "target_endpoint": endpoints[:, 0],
    }


@torch.no_grad()
def _predicted_endpoints(
    model: Any, initial: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    count, candidates, horizon = actions.shape
    flat_state = initial[:, None, :].expand(-1, candidates, -1).reshape(-1, initial.size(-1))
    flat_actions = actions.reshape(-1, horizon)
    current = flat_state
    for step in range(horizon):
        current = model(current, flat_actions[:, step])
    return current.reshape(count, candidates, -1)


@torch.no_grad()
def _evaluate_metric(
    predicted_endpoints: torch.Tensor,
    true_endpoints: torch.Tensor,
    target_endpoint: torch.Tensor,
    metric: str,
    matrix: torch.Tensor | None,
) -> dict[str, float]:
    count = true_endpoints.size(0)
    target = target_endpoint[:, None, :]
    true_difference = true_endpoints - target
    true_squared_euclidean = true_difference.square().sum(dim=-1)
    oracle_index = true_squared_euclidean.argmin(dim=1)
    oracle_distance = true_squared_euclidean.gather(1, oracle_index[:, None]).sqrt().squeeze(1)
    if metric == "oracle":
        chosen = oracle_index
    else:
        if matrix is None:
            raise ValueError(f"missing matrix for metric {metric}")
        predicted_difference = predicted_endpoints - target
        predicted_cost = torch.einsum("ncd,df,ncf->nc", predicted_difference, matrix, predicted_difference)
        chosen = predicted_cost.argmin(dim=1)
    selected_true_distance = true_squared_euclidean.gather(1, chosen[:, None]).sqrt().squeeze(1)
    return {
        "selected_goal_distance": float(selected_true_distance.mean()),
        "oracle_goal_distance": float(oracle_distance.mean()),
        "planning_regret": float((selected_true_distance - oracle_distance).mean()),
        "goal_success_rate": float((selected_true_distance <= 0.25).float().mean()),
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
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
        "negative_favors_target_for_regret": True,
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
            model, train, _test, best_epoch, best_validation, _random_scale = _fit_model(
                mode, int(seed), config
            )
            fitted[mode] = (model, train)
        matrices, diagnostics = _build_metric_matrices(fitted["none"][1], 70_000 + int(seed))
        matrix_diagnostics[str(seed)] = diagnostics
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    110_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon)
                )
                batch = _make_task_batch(tasks, candidates, horizon, generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    for metric in METRICS:
                        planning = _evaluate_metric(
                            predicted,
                            batch["true_endpoints"],
                            batch["target_endpoint"],
                            metric,
                            matrices.get(metric),
                        )
                        records.append(
                            {
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": metric,
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": planning,
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
                        and float(row["state_shift"]) == float(state_shift)
                        and int(row["planning_horizon"]) == int(horizon)
                    ]
                    if len(rows) != len(seeds):
                        raise RuntimeError("incomplete planner metric matrix")
                    summary[mode][metric][str(state_shift)][str(horizon)] = {
                        "n_seeds": len(rows),
                        "planning_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in rows])),
                        "planning_regret_std": float(np.std([r["planning"]["planning_regret"] for r in rows], ddof=1)),
                        "goal_success_rate_mean": float(np.mean([r["planning"]["goal_success_rate"] for r in rows])),
                        "oracle_choice_agreement_mean": float(np.mean([r["planning"]["oracle_choice_agreement"] for r in rows])),
                        "target_candidate_choice_rate_mean": float(np.mean([r["planning"]["target_candidate_choice_rate"] for r in rows])),
                    }

    # Primary paired inference: each learned metric against Euclidean on the
    # same seed/shift/horizon/model. Oracle is privileged and is reported but
    # excluded from the intervention family for multiplicity correction.
    paired_inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    record_map = {
        (r["model_mode"], r["metric"], float(r["state_shift"]), int(r["planning_horizon"]), int(r["seed"])): r
        for r in records
    }
    for mode in MODEL_CONDITIONS:
        for metric in METRICS[1:-1]:
            for state_shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray(
                        [
                            record_map[(mode, metric, float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                            - record_map[(mode, "euclidean", float(state_shift), int(horizon), seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    key = f"{mode}:{metric}:shift={state_shift:g}:h={horizon}"
                    paired_inference[key] = {
                        "model_mode": mode,
                        "metric": metric,
                        "state_shift": float(state_shift),
                        "planning_horizon": int(horizon),
                        **_paired_summary(deltas, seed=20261003 + int(horizon)),
                    }
                    raw_pvalues[key] = paired_inference[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        paired_inference[key]["holm_p_global"] = value

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
            "metric_estimates_use": "training transitions only",
            "oracle_is_privileged": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "matrix_diagnostics": matrix_diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": paired_inference,
        "claim_boundary": [
            "This is a nonlinear synthetic candidate-set planning proxy, not online LeWM or PushT control.",
            "The oracle metric uses true transition endpoints and is a privileged upper-bound reference.",
            "Random spectral preserves the covariance metric spectrum but randomizes its eigenbasis; it is not a nuisance-suppression proof.",
            "Metric estimates use training transitions only; test targets and true candidate endpoints are used only for evaluation.",
            "Positive or unstable effects do not support a universal claim that spectral alignment improves planning.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear planner-facing metric ablation",
        "",
        "This CPU-only experiment holds the initial states, reachable target sequences, distractor candidates, and evaluation horizons fixed across model conditions and metrics. It is a synthetic candidate-set planning proxy.",
        "",
        "## Primary results: regret",
        "",
        "| model | metric | shift | h | n | regret mean +/- sd | success | oracle agreement | target-candidate choice |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = report["summary"]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    values = summary[mode][metric][str(shift)][str(horizon)]
                    lines.append(
                        f"| {mode} | {metric} | {shift:g} | {horizon} | {values['n_seeds']} | {values['planning_regret_mean']:.6g} +/- {values['planning_regret_std']:.6g} | {values['goal_success_rate_mean']:.6g} | {values['oracle_choice_agreement_mean']:.6g} | {values['target_candidate_choice_rate_mean']:.6g} |"
                    )
    lines.extend(
        [
            "",
            "## Paired regret inference versus Euclidean",
            "",
            "The paired unit is the seed. Holm correction is across the non-oracle metric contrasts, model conditions, shifts, and horizons.",
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
    print("Running shared-candidate planner metric ablation: Euclidean, covariance, reachability, random spectral, and privileged oracle.")
    report = run_report(config, [int(seed) for seed in args.seeds], [int(horizon) for horizon in args.horizons], tasks=args.tasks, candidates=args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear planner-facing metric ablation\n\n"
        "Shared-candidate synthetic planning proxy with strict claim boundaries.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
