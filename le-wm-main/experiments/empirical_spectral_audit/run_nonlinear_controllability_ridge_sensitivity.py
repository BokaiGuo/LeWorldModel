"""Ridge-sensitivity audit for the nonlinear controllability metric.

The main controllability experiment uses a fixed Gramian regularization.  This
companion keeps each fitted model and each candidate task fixed while varying
the Gramian ridge fraction, so a negative result cannot be attributed to one
arbitrary inverse-conditioning choice.
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
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_controllability_metric_ablation import _controllability_precision, _fit_linear_transition
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import _evaluate_metric, _make_task_batch, _predicted_endpoints


MODEL_CONDITIONS = ("none", "both")
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
        "mean_delta_controllability_minus_euclidean": float(values.mean()),
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
    ridge_fractions: list[float],
    *,
    tasks: int,
    candidates: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        A, B = _fit_linear_transition(fitted["none"][1])
        diagnostics[str(seed)] = {"A": A.tolist(), "B": B.tolist(), "by_horizon": {}}
        for horizon in horizons:
            diagnostics[str(seed)]["by_horizon"][str(horizon)] = {}
            for ridge in ridge_fractions:
                _matrix, info = _controllability_precision(A, B, int(horizon), float(ridge))
                diagnostics[str(seed)]["by_horizon"][str(horizon)][str(ridge)] = info
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    150_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon)
                )
                batch = _make_task_batch(tasks, candidates, int(horizon), generator, state_shift)
                predicted_by_mode = {
                    mode: _predicted_endpoints(model, batch["initial"], batch["actions"])
                    for mode, (model, _train) in fitted.items()
                }
                for mode, predicted in predicted_by_mode.items():
                    for ridge in ridge_fractions:
                        matrix, _info = _controllability_precision(A, B, int(horizon), float(ridge))
                        planning = _evaluate_metric(
                            predicted,
                            batch["true_endpoints"],
                            batch["target_endpoint"],
                            "controllability_precision",
                            matrix,
                        )
                        records.append(
                            {
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": "controllability_precision",
                                "ridge_fraction": float(ridge),
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": planning,
                            }
                        )
                    for metric in ("euclidean", "oracle"):
                        records.append(
                            {
                                "seed": int(seed),
                                "model_mode": mode,
                                "metric": metric,
                                "ridge_fraction": 0.0,
                                "state_shift": float(state_shift),
                                "planning_horizon": int(horizon),
                                "planning": _evaluate_metric(
                                    predicted,
                                    batch["true_endpoints"],
                                    batch["target_endpoint"],
                                    metric,
                                    torch.eye(batch["target_endpoint"].size(-1)) if metric == "euclidean" else None,
                                ),
                            }
                        )

    def rows_for(mode: str, metric: str, ridge: float, shift: float, horizon: int) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["model_mode"] == mode
            and row["metric"] == metric
            and float(row["ridge_fraction"]) == float(ridge)
            and float(row["state_shift"]) == float(shift)
            and int(row["planning_horizon"]) == int(horizon)
        ]

    summary: dict[str, Any] = {}
    for mode in MODEL_CONDITIONS:
        summary[mode] = {}
        for ridge in ridge_fractions:
            summary[mode][str(ridge)] = {}
            for shift in SHIFTS:
                summary[mode][str(ridge)][str(shift)] = {}
                for horizon in horizons:
                    learned = rows_for(mode, "controllability_precision", ridge, shift, horizon)
                    baseline = rows_for(mode, "euclidean", 0.0, shift, horizon)
                    oracle = rows_for(mode, "oracle", 0.0, shift, horizon)
                    if not (len(learned) == len(baseline) == len(oracle) == len(seeds)):
                        raise RuntimeError("incomplete ridge-sensitivity matrix")
                    summary[mode][str(ridge)][str(shift)][str(horizon)] = {
                        "n_seeds": len(learned),
                        "controllability_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in learned])),
                        "controllability_regret_std": float(np.std([r["planning"]["planning_regret"] for r in learned], ddof=1)),
                        "euclidean_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in baseline])),
                        "oracle_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in oracle])),
                        "controllability_success_mean": float(np.mean([r["planning"]["goal_success_rate"] for r in learned])),
                        "euclidean_success_mean": float(np.mean([r["planning"]["goal_success_rate"] for r in baseline])),
                    }

    record_map = {
        (r["model_mode"], r["metric"], r["ridge_fraction"], r["state_shift"], r["planning_horizon"], r["seed"]): r
        for r in records
    }
    inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for ridge in ridge_fractions:
            for shift in SHIFTS:
                for horizon in horizons:
                    deltas = np.asarray(
                        [
                            record_map[(mode, "controllability_precision", ridge, shift, horizon, seed)]["planning"]["planning_regret"]
                            - record_map[(mode, "euclidean", 0.0, shift, horizon, seed)]["planning"]["planning_regret"]
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    key = f"{mode}:ridge={ridge:g}:shift={shift:g}:h={horizon}"
                    inference[key] = {
                        "model_mode": mode,
                        "ridge_fraction": float(ridge),
                        "state_shift": float(shift),
                        "planning_horizon": int(horizon),
                        **_paired_summary(deltas, seed=20261025 + int(horizon)),
                    }
                    raw_pvalues[key] = inference[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        inference[key]["holm_p_global"] = value

    report = {
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "model_conditions": list(MODEL_CONDITIONS),
            "ridge_fractions": [float(value) for value in ridge_fractions],
            "state_shifts": [float(value) for value in SHIFTS],
            "horizons": [int(value) for value in horizons],
            "planning_tasks_per_cell": int(tasks),
            "candidate_sequences_per_task": int(candidates),
            "same_fitted_models_across_ridges": True,
            "same_candidate_tasks_across_ridges": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "gramian_diagnostics": diagnostics,
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": inference,
        "claim_boundary": [
            "Ridge sensitivity tests the stability of a training-only linear Gramian inverse, not all possible nonlinear controllability estimators.",
            "The candidate-set metric is a synthetic proxy and not online LeWM or PushT control.",
            "Paired comparisons reuse the same fitted model, states, actions, targets, and candidate sets across ridge values.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear controllability ridge sensitivity",
        "",
        "The fitted models and candidate tasks are held fixed while the finite-horizon Gramian ridge fraction varies.",
        "",
        "| model | ridge | shift | h | n | Gramian regret | Euclidean regret | delta | Gramian success | Euclidean success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for ridge in report["protocol"]["ridge_fractions"]:
            for shift in SHIFTS:
                for horizon in report["protocol"]["horizons"]:
                    values = report["summary"][mode][str(ridge)][str(shift)][str(horizon)]
                    delta = values["controllability_regret_mean"] - values["euclidean_regret_mean"]
                    lines.append(
                        f"| {mode} | {ridge:g} | {shift:g} | {horizon} | {values['n_seeds']} | {values['controllability_regret_mean']:.6g} +/- {values['controllability_regret_std']:.6g} | {values['euclidean_regret_mean']:.6g} | {delta:.6g} | {values['controllability_success_mean']:.6g} | {values['euclidean_success_mean']:.6g} |"
                    )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--ridge-fractions", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
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
    print("Running paired Gramian-ridge sensitivity sweep with fixed fitted models and candidate sets.")
    report = run_report(
        config,
        [int(seed) for seed in args.seeds],
        [int(horizon) for horizon in args.horizons],
        [float(ridge) for ridge in args.ridge_fractions],
        tasks=args.tasks,
        candidates=args.candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
