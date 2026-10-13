"""Loss-weight ablation for the nonlinear multi-horizon training proxy.

The matched one-step versus multi-horizon experiment exposed a short-horizon
OOD versus long-horizon trade-off.  This follow-up holds data, initialization,
optimizer, and validation checkpoint selection fixed while varying the
relative weights on autoregressive losses at h=1,2,4.
"""

from __future__ import annotations

import argparse
import copy
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
        NonlinearWorldModel,
        _seed_everything,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import (
        EVAL_HORIZONS,
        SHIFTS,
        TrainingConfig,
        _fit_condition,
        _rollout_model,
        _sequence_metrics,
        _sequence_split,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import NonlinearWorldModel, _seed_everything
    from run_nonlinear_multihorizon_training_ablation import (
        EVAL_HORIZONS,
        SHIFTS,
        TrainingConfig,
        _fit_condition,
        _rollout_model,
        _sequence_metrics,
        _sequence_split,
    )
    from run_nonlinear_planner_metric_ablation import _evaluate_metric, _make_task_batch, _predicted_endpoints


PROFILES: dict[str, dict[int, float]] = {
    "one_step": {1: 1.0, 2: 0.0, 4: 0.0},
    "uniform": {1: 1.0, 2: 1.0, 4: 1.0},
    "short_favoring": {1: 4.0, 2: 2.0, 4: 1.0},
    "long_favoring": {1: 1.0, 2: 2.0, 4: 4.0},
    "curriculum": {1: 1.0, 2: 1.0, 4: 1.0},
}
FAMILIES = ("rollout_mse", "planning_regret", "goal_success_rate", "oracle_choice_agreement")


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


def _fit_profile(
    profile: str,
    initial_state: dict[str, torch.Tensor],
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: TrainingConfig,
) -> tuple[NonlinearWorldModel, int, float, dict[str, float]]:
    if profile == "one_step":
        return _fit_condition("one_step", initial_state, train, validation, config)
    weights = PROFILES[profile]
    model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    denominator = sum(float(weights[h]) for h in config.validation_horizons)
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predictions = _rollout_model(model, train["initial"], train["actions"])
        if profile == "curriculum":
            progress = float(epoch) / float(max(config.epochs, 1))
            scheduled_weights = {
                1: 1.0,
                2: min(1.0, max(0.0, (progress - 0.20) / 0.40)),
                4: min(1.0, max(0.0, (progress - 0.50) / 0.50)),
            }
            scheduled_denominator = sum(
                float(scheduled_weights[h]) for h in config.validation_horizons
            )
        else:
            scheduled_weights = weights
            scheduled_denominator = denominator
        objective = torch.stack(
            [
                float(scheduled_weights[h])
                * (predictions[:, h - 1] - train["targets"][:, h - 1]).square().mean()
                for h in config.validation_horizons
            ]
        ).sum() / max(scheduled_denominator, 1e-12)
        objective.backward()
        optimizer.step()
        validation_metrics = _sequence_metrics(model, validation, config.validation_horizons)
        validation_score = float(np.mean([validation_metrics[str(h)] for h in config.validation_horizons]))
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_validation_metrics = validation_metrics
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, best_validation_metrics


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
        "mean_delta_profile_minus_one_step": float(values.mean()),
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


def run_report(config: TrainingConfig, seeds: list[int]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        _seed_everything(int(seed))
        generator = torch.Generator().manual_seed(200_000 + int(seed))
        train = _sequence_split(config.train_sequences, config.sequence_horizon, generator, config.noise_std)
        validation = _sequence_split(config.validation_sequences, config.sequence_horizon, generator, 0.0)
        test = _sequence_split(config.test_sequences, config.sequence_horizon, generator, 0.0)
        initial_model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
        initial_state = copy.deepcopy(initial_model.state_dict())
        fitted: dict[str, tuple[NonlinearWorldModel, int, float, dict[str, float]]] = {}
        for profile in PROFILES:
            fitted[profile] = _fit_profile(profile, initial_state, train, validation, config)
        for profile, (model, best_epoch, validation_score, validation_metrics) in fitted.items():
            test_metrics = _sequence_metrics(model, test, config.validation_horizons)
            for horizon in EVAL_HORIZONS:
                if horizon <= config.sequence_horizon:
                    rollout = test_metrics[str(horizon)]
                else:
                    eval_generator = torch.Generator().manual_seed(205_000 + int(seed) * 10 + horizon)
                    rollout_split = _sequence_split(config.test_sequences, horizon, eval_generator, 0.0)
                    rollout = _sequence_metrics(model, rollout_split, (horizon,))[str(horizon)]
                records.append(
                    {
                        "seed": int(seed),
                        "profile": profile,
                        "family": "rollout_mse",
                        "state_shift": 0.0,
                        "horizon": int(horizon),
                        "value": float(rollout),
                        "best_epoch": int(best_epoch),
                        "validation_score": float(validation_score),
                        "validation_metrics": validation_metrics,
                    }
                )
            for shift_index, state_shift in enumerate(SHIFTS):
                for horizon in EVAL_HORIZONS:
                    task_generator = torch.Generator().manual_seed(
                        210_000 + int(seed) * 10_000 + shift_index * 1_000 + horizon
                    )
                    batch = _make_task_batch(64, 48, horizon, task_generator, state_shift)
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    planning = _evaluate_metric(
                        predicted,
                        batch["true_endpoints"],
                        batch["target_endpoint"],
                        "euclidean",
                        torch.eye(config.state_dim),
                    )
                    for family in ("planning_regret", "goal_success_rate", "oracle_choice_agreement"):
                        records.append(
                            {
                                "seed": int(seed),
                                "profile": profile,
                                "family": family,
                                "state_shift": float(state_shift),
                                "horizon": int(horizon),
                                "value": float(planning[family]),
                                "best_epoch": int(best_epoch),
                                "validation_score": float(validation_score),
                                "validation_metrics": validation_metrics,
                            }
                        )

    def select(profile: str, family: str, shift: float, horizon: int) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["profile"] == profile
            and row["family"] == family
            and float(row["state_shift"]) == float(shift)
            and int(row["horizon"]) == int(horizon)
        ]

    summary: dict[str, Any] = {}
    for profile in PROFILES:
        summary[profile] = {}
        for family in FAMILIES:
            summary[profile][family] = {}
            shifts = (0.0,) if family == "rollout_mse" else SHIFTS
            for shift in shifts:
                summary[profile][family][str(shift)] = {}
                for horizon in EVAL_HORIZONS:
                    rows = select(profile, family, shift, horizon)
                    if len(rows) != len(seeds):
                        raise RuntimeError("incomplete weight-sweep matrix")
                    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
                    summary[profile][family][str(shift)][str(horizon)] = {
                        "n_seeds": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                    }

    paired: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for profile in PROFILES:
        if profile == "one_step":
            continue
        for family in FAMILIES:
            shifts = (0.0,) if family == "rollout_mse" else SHIFTS
            for shift in shifts:
                for horizon in EVAL_HORIZONS:
                    one = {row["seed"]: row["value"] for row in select("one_step", family, shift, horizon)}
                    target = {row["seed"]: row["value"] for row in select(profile, family, shift, horizon)}
                    deltas = np.asarray([target[seed] - one[seed] for seed in seeds], dtype=np.float64)
                    key = f"{profile}:{family}:shift={shift:g}:h={horizon}"
                    paired[key] = {
                        "profile": profile,
                        "family": family,
                        "state_shift": float(shift),
                        "horizon": int(horizon),
                        **_paired_summary(deltas, seed=20261029 + horizon),
                    }
                    raw_pvalues[key] = paired[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        paired[key]["holm_p_global"] = value

    report = {
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "profiles": {name: values for name, values in PROFILES.items()},
            "families": list(FAMILIES),
            "evaluation_horizons": list(EVAL_HORIZONS),
            "state_shifts": [float(value) for value in SHIFTS],
            "same_data_and_initialization_across_profiles": True,
            "checkpoint_selection": "shared validation mean over h=1,2,4",
            "curriculum_schedule": "h=2 ramps in over epochs 0.20-0.60; h=4 ramps in over epochs 0.50-1.00",
            "planning_metric": "Euclidean predicted endpoint distance on shared candidate tasks",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": summary,
        "paired_inference_vs_one_step": paired,
        "claim_boundary": [
            "This is a synthetic loss-weight ablation, not end-to-end LeWM or PushT training.",
            "Negative deltas favor a profile for rollout MSE and planning regret; positive deltas favor it for success/agreement only when explicitly interpreted as such.",
            "The sweep tests a Pareto trade-off and does not establish a universally optimal loss weighting.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear multi-horizon loss-weight sweep",
        "",
        "All profiles share data, initialization, optimizer, validation checkpoint selection, candidate sets, and evaluation seeds.",
        "",
        "| profile | family | shift | h | n | mean +/- sd |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in PROFILES:
        for family in FAMILIES:
            shifts = (0.0,) if family == "rollout_mse" else SHIFTS
            for shift in shifts:
                for horizon in EVAL_HORIZONS:
                    values = report["summary"][profile][family][str(shift)][str(horizon)]
                    lines.append(
                        f"| {profile} | {family} | {shift:g} | {horizon} | {values['n_seeds']} | {values['mean']:.6g} +/- {values['std']:.6g} |"
                    )
    lines.extend(["", "## Paired inference versus one-step", "", "| profile | family | shift | h | n | mean delta | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
    for values in report["paired_inference_vs_one_step"].values():
        ci = values["bootstrap_ci95"]
        lines.append(
            f"| {values['profile']} | {values['family']} | {values['state_shift']:g} | {values['horizon']} | {values['n_paired_seeds']} | {values['mean_delta_profile_minus_one_step']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--train-sequences", type=int, default=512)
    parser.add_argument("--validation-sequences", type=int, default=256)
    parser.add_argument("--test-sequences", type=int, default=1024)
    parser.add_argument("--sequence-horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--noise-std", type=float, default=0.02)
    args = parser.parse_args()
    config = TrainingConfig(
        train_sequences=args.train_sequences,
        validation_sequences=args.validation_sequences,
        test_sequences=args.test_sequences,
        sequence_horizon=args.sequence_horizon,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
    )
    print("Running matched multi-horizon loss-weight sweep: one-step, uniform, short-favoring, long-favoring, curriculum.")
    report = run_report(config, [int(seed) for seed in args.seeds])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
