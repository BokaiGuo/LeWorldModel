"""One-step versus multi-horizon training ablation.

The post-hoc planner metrics did not repair predicted endpoint error.  This
experiment tests the training-side proposal directly: keep the same predictor,
data, initialization, optimizer, and checkpoint rule, but compare one-step
loss with an autoregressive multi-horizon loss.  Evaluation includes held-out
one-step error, autoregressive rollout, and shared-candidate planning.

This is a nonlinear synthetic training proxy, not an end-to-end LeWM or PushT
training result.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        NonlinearWorldModel,
        _seed_everything,
        true_transition,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planner_metric_ablation import (
        _evaluate_metric,
        _make_task_batch,
        _predicted_endpoints,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        NonlinearWorldModel,
        _seed_everything,
        true_transition,
    )
    from run_nonlinear_planner_metric_ablation import _evaluate_metric, _make_task_batch, _predicted_endpoints


CONDITIONS = ("one_step", "multi_horizon")
EVAL_HORIZONS = (1, 2, 4, 8)
SHIFTS = (0.0, 1.5)


@dataclass(frozen=True)
class TrainingConfig:
    state_dim: int = 4
    hidden_dim: int = 64
    train_sequences: int = 512
    validation_sequences: int = 256
    test_sequences: int = 1024
    sequence_horizon: int = 4
    epochs: int = 160
    learning_rate: float = 2e-3
    noise_std: float = 0.02
    planning_tasks: int = 64
    planning_candidates: int = 48
    validation_horizons: tuple[int, ...] = (1, 2, 4)


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


def _sequence_split(
    count: int,
    horizon: int,
    generator: torch.Generator,
    noise_std: float,
    *,
    state_shift: float = 0.0,
) -> dict[str, torch.Tensor]:
    initial = torch.randn(count, 4, generator=generator) + float(state_shift)
    actions = (torch.rand(count, horizon, generator=generator) < 0.5).long()
    states = [initial]
    current = initial
    for step in range(horizon):
        current = true_transition(current, actions[:, step])
        states.append(current)
    clean_targets = torch.stack(states[1:], dim=1)
    targets = clean_targets.clone()
    if noise_std > 0:
        targets = targets + float(noise_std) * torch.randn(targets.shape, generator=generator)
    return {
        "initial": initial,
        "actions": actions,
        "targets": targets,
        "clean_targets": clean_targets,
    }


def _rollout_model(
    model: NonlinearWorldModel,
    initial: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    current = initial
    predictions: list[torch.Tensor] = []
    for step in range(actions.size(1)):
        current = model(current, actions[:, step])
        predictions.append(current)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def _sequence_metrics(
    model: NonlinearWorldModel,
    split: dict[str, torch.Tensor],
    horizons: tuple[int, ...],
) -> dict[str, float]:
    predictions = _rollout_model(model, split["initial"], split["actions"])
    output: dict[str, float] = {}
    for horizon in horizons:
        if horizon > predictions.size(1):
            continue
        output[str(horizon)] = float(
            (predictions[:, horizon - 1] - split["clean_targets"][:, horizon - 1]).square().mean()
        )
    return output


def _fit_condition(
    condition: str,
    initial_state: dict[str, torch.Tensor],
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: TrainingConfig,
) -> tuple[NonlinearWorldModel, int, float, dict[str, float]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = math.inf
    best_epoch = 0
    best_validation_metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predictions = _rollout_model(model, train["initial"], train["actions"])
        if condition == "one_step":
            objective = (predictions[:, 0] - train["targets"][:, 0]).square().mean()
        else:
            objective = torch.stack(
                [
                    (predictions[:, horizon - 1] - train["targets"][:, horizon - 1]).square().mean()
                    for horizon in config.validation_horizons
                ]
            ).mean()
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


def _rollout_metrics(
    model: NonlinearWorldModel,
    count: int,
    generator: torch.Generator,
    horizon: int,
    state_shift: float,
) -> dict[str, float]:
    split = _sequence_split(count, horizon, generator, 0.0, state_shift=state_shift)
    return _sequence_metrics(model, split, tuple(value for value in EVAL_HORIZONS if value <= horizon))


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
        "mean_delta_multi_horizon_minus_one_step": float(values.mean()),
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
        generator = torch.Generator().manual_seed(180_000 + int(seed))
        train = _sequence_split(config.train_sequences, config.sequence_horizon, generator, config.noise_std)
        validation = _sequence_split(config.validation_sequences, config.sequence_horizon, generator, 0.0)
        test = _sequence_split(config.test_sequences, config.sequence_horizon, generator, 0.0)
        initial_model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
        initial_state = copy.deepcopy(initial_model.state_dict())
        fitted: dict[str, tuple[NonlinearWorldModel, int, float, dict[str, float]]] = {}
        for condition in CONDITIONS:
            fitted[condition] = _fit_condition(condition, initial_state, train, validation, config)
        for condition, (model, best_epoch, validation_score, validation_metrics) in fitted.items():
            test_metrics = _sequence_metrics(model, test, tuple(value for value in config.validation_horizons))
            for horizon in EVAL_HORIZONS:
                if horizon <= config.sequence_horizon:
                    rollout = test_metrics.get(str(horizon), float("nan"))
                else:
                    eval_generator = torch.Generator().manual_seed(185_000 + int(seed) * 10 + horizon)
                    rollout = _rollout_metrics(model, config.test_sequences, eval_generator, horizon, 0.0)[str(horizon)]
                records.append(
                    {
                        "seed": int(seed),
                        "condition": condition,
                        "metric_family": "rollout_mse",
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
                    generator = torch.Generator().manual_seed(
                        190_000 + int(seed) * 10_000 + shift_index * 1_000 + horizon
                    )
                    batch = _make_task_batch(
                        config.planning_tasks,
                        config.planning_candidates,
                        horizon,
                        generator,
                        state_shift,
                    )
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    planning = _evaluate_metric(
                        predicted,
                        batch["true_endpoints"],
                        batch["target_endpoint"],
                        "euclidean",
                        torch.eye(config.state_dim),
                    )
                    for metric_name in ("planning_regret", "goal_success_rate", "oracle_choice_agreement"):
                        records.append(
                            {
                                "seed": int(seed),
                                "condition": condition,
                                "metric_family": metric_name,
                                "state_shift": float(state_shift),
                                "horizon": int(horizon),
                                "value": float(planning[metric_name]),
                                "best_epoch": int(best_epoch),
                                "validation_score": float(validation_score),
                                "validation_metrics": validation_metrics,
                            }
                        )

    def select(condition: str, family: str, shift: float, horizon: int) -> list[dict[str, Any]]:
        return [
            row
            for row in records
            if row["condition"] == condition
            and row["metric_family"] == family
            and float(row["state_shift"]) == float(shift)
            and int(row["horizon"]) == int(horizon)
        ]

    summary: dict[str, Any] = {}
    families = ("rollout_mse", "planning_regret", "goal_success_rate", "oracle_choice_agreement")
    for condition in CONDITIONS:
        summary[condition] = {}
        for family in families:
            summary[condition][family] = {}
            shifts = (0.0,) if family == "rollout_mse" else SHIFTS
            for shift in shifts:
                summary[condition][family][str(shift)] = {}
                for horizon in EVAL_HORIZONS:
                    rows = select(condition, family, shift, horizon)
                    if len(rows) != len(seeds):
                        raise RuntimeError(f"incomplete record matrix for {condition}/{family}/{shift}/{horizon}")
                    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
                    summary[condition][family][str(shift)][str(horizon)] = {
                        "n_seeds": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                    }

    paired: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for family in families:
        shifts = (0.0,) if family == "rollout_mse" else SHIFTS
        for shift in shifts:
            for horizon in EVAL_HORIZONS:
                one = {(row["seed"]): row["value"] for row in select("one_step", family, shift, horizon)}
                multi = {(row["seed"]): row["value"] for row in select("multi_horizon", family, shift, horizon)}
                deltas = np.asarray([multi[seed] - one[seed] for seed in seeds], dtype=np.float64)
                key = f"{family}:shift={shift:g}:h={horizon}"
                paired[key] = {
                    "metric_family": family,
                    "state_shift": float(shift),
                    "horizon": int(horizon),
                    **_paired_summary(deltas, seed=20261028 + horizon),
                }
                raw_pvalues[key] = paired[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        paired[key]["holm_p_global"] = value

    report = {
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in seeds],
            "conditions": list(CONDITIONS),
            "evaluation_horizons": list(EVAL_HORIZONS),
            "state_shifts": [float(value) for value in SHIFTS],
            "same_data_and_initialization_across_conditions": True,
            "checkpoint_selection": "shared validation mean over h=1,2,4",
            "multi_horizon_objective": "autoregressive rollout MSE over h=1,2,4",
            "planning_metric": "Euclidean predicted endpoint distance; true endpoint used only for scoring",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": summary,
        "paired_inference_multi_horizon_minus_one_step": paired,
        "claim_boundary": [
            "This is a nonlinear synthetic autoregressive training ablation, not end-to-end LeWM or PushT training.",
            "The multi-horizon objective backpropagates through predicted states; it is not teacher-forced beyond the initial state.",
            "Planning uses a shared candidate-set proxy and does not establish online control success.",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear one-step versus multi-horizon training ablation",
        "",
        "Both conditions use the same synthetic sequences, initialization, optimizer, and shared validation checkpoint rule. The multi-horizon condition backpropagates through autoregressive predictions at h=1,2,4.",
        "",
        "| condition | family | shift | h | n | mean +/- sd |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        for family in ("rollout_mse", "planning_regret", "goal_success_rate", "oracle_choice_agreement"):
            shifts = (0.0,) if family == "rollout_mse" else SHIFTS
            for shift in shifts:
                for horizon in EVAL_HORIZONS:
                    values = report["summary"][condition][family][str(shift)][str(horizon)]
                    lines.append(
                        f"| {condition} | {family} | {shift:g} | {horizon} | {values['n_seeds']} | {values['mean']:.6g} +/- {values['std']:.6g} |"
                    )
    lines.extend(["", "## Paired inference", "", "| family | shift | h | n | mean delta multi - one-step | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
    for values in report["paired_inference_multi_horizon_minus_one_step"].values():
        ci = values["bootstrap_ci95"]
        lines.append(
            f"| {values['metric_family']} | {values['state_shift']:g} | {values['horizon']} | {values['n_paired_seeds']} | {values['mean_delta_multi_horizon_minus_one_step']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |"
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
    parser.add_argument("--planning-tasks", type=int, default=64)
    parser.add_argument("--planning-candidates", type=int, default=48)
    args = parser.parse_args()
    config = TrainingConfig(
        train_sequences=args.train_sequences,
        validation_sequences=args.validation_sequences,
        test_sequences=args.test_sequences,
        sequence_horizon=args.sequence_horizon,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        planning_tasks=args.planning_tasks,
        planning_candidates=args.planning_candidates,
    )
    print("Running matched one-step versus autoregressive multi-horizon training ablation.")
    report = run_report(config, [int(seed) for seed in args.seeds])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
