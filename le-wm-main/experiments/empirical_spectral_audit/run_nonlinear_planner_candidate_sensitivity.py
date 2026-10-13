"""Candidate-set sensitivity for the nonlinear planner metric ablation.

This companion reuses the same fitted models and training-only metric
estimates while varying the number of candidate action sequences.  The target
sequence remains candidate zero, so every candidate set contains a reachable
goal and the oracle reference is well-defined.  It tests whether metric
comparisons survive a change in search-set size rather than only one arbitrary
48-candidate configuration.
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
        METRICS,
        SHIFTS,
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model
    from run_nonlinear_planner_metric_ablation import (
        METRICS,
        SHIFTS,
        _build_metric_matrices,
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )


MODEL_CONDITIONS = ("none", "both")


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
    candidate_counts: list[int],
    *,
    tasks: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        fitted: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}
        for mode in MODEL_CONDITIONS:
            model, train, _test, _epoch, _validation, _scale = _fit_model(mode, int(seed), config)
            fitted[mode] = (model, train)
        matrices, _diagnostics = _build_metric_matrices(fitted["none"][1], 75_000 + int(seed))
        for shift_index, state_shift in enumerate(SHIFTS):
            for horizon in horizons:
                for count_index, candidates in enumerate(candidate_counts):
                    generator = torch.Generator().manual_seed(
                        125_000 + int(seed) * 10_000 + shift_index * 1_000 + int(horizon) * 10 + count_index
                    )
                    batch = _make_task_batch(tasks, candidates, horizon, generator, state_shift)
                    for mode, (model, _train) in fitted.items():
                        predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                        for metric in METRICS:
                            records.append(
                                {
                                    "seed": int(seed),
                                    "model_mode": mode,
                                    "metric": metric,
                                    "state_shift": float(state_shift),
                                    "planning_horizon": int(horizon),
                                    "candidate_count": int(candidates),
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
            for candidates in candidate_counts:
                summary[mode][metric][str(candidates)] = {}
                for state_shift in SHIFTS:
                    summary[mode][metric][str(candidates)][str(state_shift)] = {}
                    for horizon in horizons:
                        rows = [
                            row
                            for row in records
                            if row["model_mode"] == mode
                            and row["metric"] == metric
                            and row["candidate_count"] == candidates
                            and row["state_shift"] == state_shift
                            and row["planning_horizon"] == horizon
                        ]
                        if len(rows) != len(seeds):
                            raise RuntimeError("incomplete candidate-sensitivity matrix")
                        summary[mode][metric][str(candidates)][str(state_shift)][str(horizon)] = {
                            "n_seeds": len(rows),
                            "planning_regret_mean": float(np.mean([r["planning"]["planning_regret"] for r in rows])),
                            "planning_regret_std": float(np.std([r["planning"]["planning_regret"] for r in rows], ddof=1)),
                            "goal_success_rate_mean": float(np.mean([r["planning"]["goal_success_rate"] for r in rows])),
                            "oracle_choice_agreement_mean": float(np.mean([r["planning"]["oracle_choice_agreement"] for r in rows])),
                        }

    record_map = {
        (r["model_mode"], r["metric"], r["candidate_count"], r["state_shift"], r["planning_horizon"], r["seed"]): r
        for r in records
    }
    inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in MODEL_CONDITIONS:
        for metric in METRICS[1:-1]:
            for candidates in candidate_counts:
                for state_shift in SHIFTS:
                    for horizon in horizons:
                        deltas = np.asarray(
                            [
                                record_map[(mode, metric, candidates, state_shift, horizon, seed)]["planning"]["planning_regret"]
                                - record_map[(mode, "euclidean", candidates, state_shift, horizon, seed)]["planning"]["planning_regret"]
                                for seed in seeds
                            ],
                            dtype=np.float64,
                        )
                        key = f"{mode}:{metric}:c={candidates}:shift={state_shift:g}:h={horizon}"
                        inference[key] = {
                            "model_mode": mode,
                            "metric": metric,
                            "candidate_count": int(candidates),
                            "state_shift": float(state_shift),
                            "planning_horizon": int(horizon),
                            **_paired_summary(deltas, seed=20261017 + candidates + horizon),
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
            "candidate_counts": [int(value) for value in candidate_counts],
            "planning_tasks_per_cell": int(tasks),
            "candidate_zero_is_reachable_target_sequence": True,
            "metric_estimates_use": "training transitions only",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": summary,
        "paired_regret_inference_vs_euclidean": inference,
        "claim_boundary": [
            "This is candidate-set sensitivity for a nonlinear synthetic proxy, not online LeWM or PushT control.",
            "Candidate-count comparisons use independent but deterministic shared-protocol task batches per cell.",
            "The random spectral and oracle controls retain the boundaries declared in the primary metric ablation.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear planner candidate-set sensitivity",
        "",
        "Metric comparisons are repeated at multiple candidate-set sizes. Values are mean +/- seed standard deviation over the shared synthetic task protocol.",
        "",
        "| model | metric | candidates | shift | h | n | regret mean +/- sd | success | oracle agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODEL_CONDITIONS:
        for metric in METRICS:
            for candidates in report["protocol"]["candidate_counts"]:
                for shift in SHIFTS:
                    for horizon in report["protocol"]["horizons"]:
                        values = report["summary"][mode][metric][str(candidates)][str(shift)][str(horizon)]
                        lines.append(
                            f"| {mode} | {metric} | {candidates} | {shift:g} | {horizon} | {values['n_seeds']} | {values['planning_regret_mean']:.6g} +/- {values['planning_regret_std']:.6g} | {values['goal_success_rate_mean']:.6g} | {values['oracle_choice_agreement_mean']:.6g} |"
                        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[16, 48, 96])
    parser.add_argument("--tasks", type=int, default=64)
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
    print("Running candidate-set sensitivity at 16, 48, and 96 sequences with shared nonlinear training conditions.")
    report = run_report(
        config,
        [int(seed) for seed in args.seeds],
        [int(horizon) for horizon in args.horizons],
        [int(count) for count in args.candidate_counts],
        tasks=args.tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear planner candidate-set sensitivity\n\n"
        "Candidate-count robustness companion to the planner metric ablation.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
