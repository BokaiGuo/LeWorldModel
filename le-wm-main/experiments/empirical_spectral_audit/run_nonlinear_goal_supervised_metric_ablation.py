"""Goal-conditioned reachability metric ablation.

The dynamics-only metrics can identify predictive or action-sensitive
directions without identifying which endpoint errors matter to a task.  This
controlled synthetic experiment adds an explicit goal-reaching task: only the
first two state coordinates define task distance, while the remaining two are
task-irrelevant nuisance coordinates.  A positive diagonal planner metric is
learned from independent success/failure candidate tasks and evaluated on
held-out tasks.

This tests the proposal's reachability-supervised planner metric idea.  It is
synthetic task-supervision evidence, not a LeWM/PushT result.
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
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import _build_metric_matrices, _make_task_batch, _predicted_endpoints


MODEL_CONDITIONS = ("none", "global", "conditional", "predictor", "both")
METRICS = ("euclidean", "covariance_precision", "task_supervised", "random_spectral", "oracle")
SHIFTS = (0.0, 1.5)
TASK_WEIGHTS = torch.tensor([4.0, 1.0, 0.15, 0.15], dtype=torch.float32)
SUCCESS_THRESHOLD = 0.25


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


def _squared_features(endpoints: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (endpoints - targets[:, None, :]).square()


def _fit_task_metric(
    tasks: dict[str, torch.Tensor],
    *,
    steps: int = 400,
    learning_rate: float = 5e-2,
    l2: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit positive diagonal weights from success/failure labels only."""

    features = _squared_features(tasks["true_endpoints"], tasks["target_endpoint"]).reshape(-1, 4)
    true_task_distance = torch.sqrt(features @ TASK_WEIGHTS).reshape(-1)
    labels = (true_task_distance <= SUCCESS_THRESHOLD).float()
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    positive_weight = negative / positive
    log_weights = torch.nn.Parameter(torch.zeros(4))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([log_weights, bias], lr=learning_rate)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.nn.functional.softplus(log_weights)
        logits = bias - features @ weights
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=positive_weight,
        ) + float(l2) * weights.square().mean()
        loss.backward()
        optimizer.step()
    weights = torch.nn.functional.softplus(log_weights).detach()
    weights = weights / weights.mean().clamp_min(1e-8)
    normalized_true = TASK_WEIGHTS / TASK_WEIGHTS.mean()
    cosine = float(torch.dot(weights, normalized_true) / (weights.norm() * normalized_true.norm()).clamp_min(1e-8))
    return torch.diag(weights), {
        "weights": [float(value) for value in weights],
        "normalized_true_task_weights": [float(value) for value in normalized_true],
        "weight_cosine_to_true_task": cosine,
        "training_candidates": int(features.size(0)),
        "positive_labels": int(labels.sum()),
        "negative_labels": int((1.0 - labels).sum()),
        "success_threshold": float(SUCCESS_THRESHOLD),
        "fit_target": "success/failure only; no task distance supplied to optimizer",
    }


@torch.no_grad()
def _evaluate_metric(
    predicted_endpoints: torch.Tensor,
    true_endpoints: torch.Tensor,
    target_endpoint: torch.Tensor,
    metric: str,
    matrix: torch.Tensor | None,
) -> dict[str, float]:
    target = target_endpoint[:, None, :]
    true_difference = true_endpoints - target
    true_task_cost = true_difference.square() @ TASK_WEIGHTS
    oracle_index = true_task_cost.argmin(dim=1)
    oracle_distance = true_task_cost.gather(1, oracle_index[:, None]).sqrt().squeeze(1)
    if metric == "oracle":
        chosen = oracle_index
    else:
        if matrix is None:
            raise ValueError(f"missing metric matrix: {metric}")
        predicted_difference = predicted_endpoints - target
        predicted_cost = torch.einsum("ncd,df,ncf->nc", predicted_difference, matrix, predicted_difference)
        chosen = predicted_cost.argmin(dim=1)
    selected_distance = true_task_cost.gather(1, chosen[:, None]).sqrt().squeeze(1)
    return {
        "selected_task_distance": float(selected_distance.mean()),
        "oracle_task_distance": float(oracle_distance.mean()),
        "planning_regret": float((selected_distance - oracle_distance).mean()),
        "task_success_rate": float((selected_distance <= SUCCESS_THRESHOLD).float().mean()),
        "oracle_success_rate": float((oracle_distance <= SUCCESS_THRESHOLD).float().mean()),
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
    metric_train_tasks: int,
    metric_train_candidates: int,
    fit_steps: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    metric_diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        base_matrices, base_diagnostics = _build_metric_matrices(fitted["none"][1], 95_000 + int(seed))
        task_matrices: dict[int, torch.Tensor] = {}
        task_infos: dict[str, Any] = {}
        for horizon in horizons:
            metric_generator = torch.Generator().manual_seed(
                170_000 + int(seed) * 1_000 + int(horizon)
            )
            metric_tasks = _make_task_batch(
                metric_train_tasks,
                metric_train_candidates,
                int(horizon),
                metric_generator,
                0.0,
            )
            task_matrices[int(horizon)], task_infos[str(horizon)] = _fit_task_metric(
                metric_tasks, steps=fit_steps
            )
        metric_diagnostics[str(seed)] = {
            "task_metric_by_horizon": task_infos,
            "base_metric": base_diagnostics,
        }
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    175_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon)
                )
                batch = _make_task_batch(tasks, candidates, int(horizon), generator, state_shift)
                for mode, (model, _train) in fitted.items():
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    matrices = dict(base_matrices)
                    matrices["task_supervised"] = task_matrices[int(horizon)]
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
                        raise RuntimeError("incomplete goal-supervised metric matrix")
                    summary[mode][metric][str(shift)][str(horizon)] = {
                        "n_seeds": len(rows),
                        "planning_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in rows])),
                        "planning_regret_std": float(np.std([r["planning"]["planning_regret"] for r in rows], ddof=1)),
                        "task_success_rate_mean": float(np.mean([r["planning"]["task_success_rate"] for r in rows])),
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
                        **_paired_summary(deltas, seed=20261027 + int(horizon)),
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
            "metric_training_tasks_per_horizon": int(metric_train_tasks),
            "metric_training_candidates": int(metric_train_candidates),
            "metric_fit_steps": int(fit_steps),
            "horizon_conditioned_task_metric": True,
            "same_task_metric_across_models": True,
            "evaluation_uses_held_out_tasks": True,
            "synthetic_only": True,
            "task_definition": "weighted goal distance on coordinates 0,1; coordinates 2,3 are task-irrelevant",
            "task_weights": [float(value) for value in TASK_WEIGHTS],
            "success_threshold": float(SUCCESS_THRESHOLD),
            "elapsed_seconds": time.time() - started,
        },
        "metric_diagnostics": metric_diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": inference,
        "claim_boundary": [
            "This is controlled synthetic goal-conditioned supervision, not a physical nuisance annotation or LeWM/PushT result.",
            "The task metric is learned from independent success/failure candidate labels and evaluated on held-out candidate tasks.",
            "The oracle is privileged and uses true task distances; learned task metrics use predicted endpoints at evaluation.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear goal-supervised reachability metric ablation",
        "",
        "A positive diagonal goal metric is learned from independent success/failure candidate labels. Evaluation uses held-out candidate tasks and a weighted goal distance whose first two coordinates are task-relevant.",
        "",
        "| model | metric | shift | h | n | regret mean +/- sd | task success | oracle agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    values = report["summary"][mode][metric][str(shift)][str(horizon)]
                    lines.append(
                        f"| {mode} | {metric} | {shift:g} | {horizon} | {values['n_seeds']} | {values['planning_regret_mean']:.6g} +/- {values['planning_regret_std']:.6g} | {values['task_success_rate_mean']:.6g} | {values['oracle_choice_agreement_mean']:.6g} |"
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
    parser.add_argument("--metric-train-tasks", type=int, default=256)
    parser.add_argument("--metric-train-candidates", type=int, default=64)
    parser.add_argument("--fit-steps", type=int, default=400)
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
    print("Running goal-supervised reachability metric ablation with held-out candidate tasks.")
    report = run_report(
        config,
        [int(seed) for seed in args.seeds],
        [int(horizon) for horizon in args.horizons],
        tasks=args.tasks,
        candidates=args.candidates,
        metric_train_tasks=args.metric_train_tasks,
        metric_train_candidates=args.metric_train_candidates,
        fit_steps=args.fit_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
