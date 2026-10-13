"""Functional consequences of conditional alignment interventions.

This benchmark extends the learned action-conditioned alignment experiment with
random action-sequence rollouts and a controlled candidate-set planning proxy.
It keeps the data-generating process, optimizer, and alignment loss fixed while
evaluating whether global or action-conditional alignment changes function.

The artifact is a CPU-only synthetic mechanism benchmark.  The planning
metric is not online task success: an independent true target trajectory
defines the goal, the model selects using its predicted final state, and the
selected physical distance is compared with the oracle candidate set.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from experiments.empirical_spectral_audit.regularizers import operator_alignment_loss
    from experiments.empirical_spectral_audit.run_alignment_training_synthetic import (
        ActionConditionedLinear,
        BenchmarkConfig,
        _action_composition_errors,
        _diagnostics,
        _make_ground_truth,
        _mse_by_group,
        _predict,
        _sample_split,
    )
except ModuleNotFoundError:  # direct invocation from the project root or package snapshot
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    try:
        from experiments.empirical_spectral_audit.regularizers import operator_alignment_loss
        from experiments.empirical_spectral_audit.run_alignment_training_synthetic import (
            ActionConditionedLinear,
            BenchmarkConfig,
            _action_composition_errors,
            _diagnostics,
            _make_ground_truth,
            _mse_by_group,
            _predict,
            _sample_split,
        )
    except ModuleNotFoundError:  # standalone reproducibility-package execution
        from regularizers import operator_alignment_loss
        from run_alignment_training_synthetic import (
            ActionConditionedLinear,
            BenchmarkConfig,
            _action_composition_errors,
            _diagnostics,
            _make_ground_truth,
            _mse_by_group,
            _predict,
            _sample_split,
        )


@dataclass(frozen=True)
class FunctionalAlignmentConfig:
    train_samples_per_group: int = 512
    validation_samples_per_group: int = 512
    test_samples_per_group: int = 2048
    epochs: int = 300
    learning_rate: float = 0.05
    ridge: float = 1e-3
    min_samples: int = 32
    alignment_bins: int = 2
    noise_std: float = 0.02
    rollout_trajectories: int = 256
    rollout_horizon: int = 8
    planning_tasks: int = 96
    planning_candidates: int = 64
    planning_horizon: int = 4
    planning_success_threshold: float = 0.25
    ood_group_one_probability: float = 0.80
    ood_state_shift: float = 1.0


@dataclass
class TrainedAlignmentBenchmark:
    model: ActionConditionedLinear
    true_operators: torch.Tensor
    train: dict[str, torch.Tensor]
    validation: dict[str, torch.Tensor]
    test: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_mse: float


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _base_config(config: FunctionalAlignmentConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        train_samples_per_group=config.train_samples_per_group,
        validation_samples_per_group=config.validation_samples_per_group,
        test_samples_per_group=config.test_samples_per_group,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        ridge=config.ridge,
        min_samples=config.min_samples,
        alignment_bins=config.alignment_bins,
        noise_std=config.noise_std,
        rollout_horizons=(1, 2, 4, 8),
    )


def _fit_model(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: FunctionalAlignmentConfig,
) -> TrainedAlignmentBenchmark:
    if mode not in {"none", "global", "conditional"}:
        raise ValueError(f"unknown alignment mode: {mode}")
    _seed_everything(seed)
    device = torch.device("cpu")
    base_config = _base_config(config)
    data_generator = torch.Generator().manual_seed(seed)
    model_generator = torch.Generator().manual_seed(seed + 10_000)
    true_covariances, true_operators = _make_ground_truth(base_config.dim, device)
    train = _sample_split(
        base_config.train_samples_per_group,
        true_covariances,
        true_operators,
        base_config.noise_std,
        data_generator,
    )
    validation = _sample_split(
        base_config.validation_samples_per_group,
        true_covariances,
        true_operators,
        base_config.noise_std,
        data_generator,
    )
    test = _sample_split(
        base_config.test_samples_per_group,
        true_covariances,
        true_operators,
        0.0,
        data_generator,
    )
    model = ActionConditionedLinear(base_config.dim, model_generator, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_mse = math.inf
    for epoch in range(1, base_config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, train)
        prediction_mse = (prediction - train["target"]).square().mean()
        alignment = operator_alignment_loss(
            train["source"],
            prediction,
            actions=train["action"],
            mode=mode,
            num_bins=base_config.alignment_bins,
            min_samples=base_config.min_samples,
            ridge=base_config.ridge,
        )
        objective = prediction_mse + float(alignment_weight) * alignment
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            validation_mse = float(
                (_predict(model, validation) - validation["target"])
                .square()
                .mean()
            )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state)
    return TrainedAlignmentBenchmark(
        model=model,
        true_operators=true_operators,
        train=train,
        validation=validation,
        test=test,
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
    )


def _sample_sequence_split(
    count: int,
    horizon: int,
    true_covariances: torch.Tensor,
    true_operators: torch.Tensor,
    generator: torch.Generator,
    *,
    group_one_probability: float = 0.5,
    state_shift: float = 0.0,
) -> dict[str, torch.Tensor]:
    dim = true_covariances.size(-1)
    initial_groups = (
        torch.rand(count, generator=generator) < group_one_probability
    ).long()
    initial = torch.empty(count, dim)
    for group in (0, 1):
        mask = initial_groups == group
        if bool(mask.any()):
            chol = torch.linalg.cholesky(true_covariances[group])
            initial[mask] = (
                torch.randn(int(mask.sum()), dim, generator=generator) @ chol.T
            )
    initial = initial + state_shift
    groups = (
        torch.rand(count, horizon, generator=generator) < group_one_probability
    ).long()
    states = [initial]
    current = initial
    for step in range(horizon):
        current = torch.bmm(
            true_operators[groups[:, step]], current.unsqueeze(-1)
        ).squeeze(-1)
        states.append(current)
    return {"states": torch.stack(states, dim=1), "groups": groups}


@torch.no_grad()
def _random_rollout_mse(
    model: ActionConditionedLinear,
    split: dict[str, torch.Tensor],
    true_operators: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    states = split["states"]
    groups = split["groups"]
    current = states[:, 0]
    result: dict[str, float] = {}
    for step in range(groups.size(1)):
        action = groups[:, step].float().unsqueeze(-1)
        current = model(current, action)
        if step + 1 in horizons:
            result[str(step + 1)] = float(
                (current - states[:, step + 1]).square().mean()
            )
    return result


@torch.no_grad()
def _planning_proxy(
    model: ActionConditionedLinear,
    true_operators: torch.Tensor,
    config: FunctionalAlignmentConfig,
    generator: torch.Generator,
    *,
    group_one_probability: float,
    state_shift: float,
) -> dict[str, float]:
    """Select candidates by predicted state distance and score physical distance."""

    state_dim = true_operators.size(-1)
    initial_groups = (
        torch.rand(config.planning_tasks, generator=generator)
        < group_one_probability
    ).long()
    initial = torch.empty(config.planning_tasks, state_dim)
    covariances, _ = _make_ground_truth(state_dim, torch.device("cpu"))
    for group in (0, 1):
        mask = initial_groups == group
        if bool(mask.any()):
            chol = torch.linalg.cholesky(covariances[group])
            initial[mask] = (
                torch.randn(int(mask.sum()), state_dim, generator=generator) @ chol.T
            )
    initial = initial + state_shift
    groups = (
        torch.rand(
            config.planning_tasks,
            config.planning_candidates,
            config.planning_horizon,
            generator=generator,
        )
        < group_one_probability
    ).long()
    target_groups = (
        torch.rand(
            config.planning_tasks,
            config.planning_horizon,
            generator=generator,
        )
        < group_one_probability
    ).long()
    target = initial.clone()
    for step in range(config.planning_horizon):
        target = torch.bmm(
            true_operators[target_groups[:, step]], target.unsqueeze(-1)
        ).squeeze(-1)
    true_current = initial[:, None, :].expand(
        -1, config.planning_candidates, -1
    ).reshape(-1, state_dim)
    predicted_current = true_current.clone()
    for step in range(config.planning_horizon):
        action = groups[:, :, step].reshape(-1, 1).float()
        true_current = torch.bmm(
            true_operators[groups[:, :, step].reshape(-1)],
            true_current.unsqueeze(-1),
        ).squeeze(-1)
        predicted_current = model(predicted_current, action)
    true_final = true_current.reshape(
        config.planning_tasks, config.planning_candidates, state_dim
    )
    physical_distance = (true_final - target[:, None, :]).norm(dim=-1)
    oracle_index = physical_distance.argmin(dim=1)
    oracle_distance = physical_distance.gather(1, oracle_index[:, None]).squeeze(1)
    predicted_final = predicted_current.reshape(
        config.planning_tasks, config.planning_candidates, state_dim
    )
    predicted_distance = (predicted_final - target[:, None, :]).norm(dim=-1)
    selected_index = predicted_distance.argmin(dim=1)
    selected_distance = physical_distance.gather(1, selected_index[:, None]).squeeze(1)
    return {
        "selected_goal_distance": float(selected_distance.mean()),
        "oracle_goal_distance": float(oracle_distance.mean()),
        "planning_regret": float((selected_distance - oracle_distance).mean()),
        "goal_success_rate": float(
            (selected_distance <= config.planning_success_threshold).float().mean()
        ),
        "oracle_success_rate": float(
            (oracle_distance <= config.planning_success_threshold).float().mean()
        ),
        "oracle_choice_agreement": float(
            (selected_index == oracle_index).float().mean()
        ),
    }


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def run_one(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: FunctionalAlignmentConfig,
) -> dict[str, object]:
    bundle = _fit_model(mode, alignment_weight, seed, config)
    generator = torch.Generator().manual_seed(100_000 + seed)
    id_rollout = _sample_sequence_split(
        config.rollout_trajectories,
        config.rollout_horizon,
        *_make_ground_truth(4, torch.device("cpu")),
        generator,
    )
    ood_rollout = _sample_sequence_split(
        config.rollout_trajectories,
        config.rollout_horizon,
        *_make_ground_truth(4, torch.device("cpu")),
        generator,
        group_one_probability=config.ood_group_one_probability,
        state_shift=config.ood_state_shift,
    )
    id_plan = _planning_proxy(
        bundle.model,
        bundle.true_operators,
        config,
        generator,
        group_one_probability=0.5,
        state_shift=0.0,
    )
    ood_plan = _planning_proxy(
        bundle.model,
        bundle.true_operators,
        config,
        generator,
        group_one_probability=config.ood_group_one_probability,
        state_shift=config.ood_state_shift,
    )
    with torch.no_grad():
        error = _predict(bundle.model, bundle.test) - bundle.test["target"]
        record: dict[str, object] = {
            "mode": mode,
            "alignment_weight": alignment_weight,
            "seed": seed,
            "device": "cpu",
            "best_epoch": bundle.best_epoch,
            "best_validation_mse": bundle.best_validation_mse,
            "test_one_step_mse": float(error.square().mean()),
            "test_one_step_mse_by_group": _mse_by_group(error, bundle.test["action"]),
            "random_rollout_mse": {
                "id": _random_rollout_mse(
                    bundle.model, id_rollout, bundle.true_operators, (1, 2, 4, 8)
                ),
                "ood": _random_rollout_mse(
                    bundle.model, ood_rollout, bundle.true_operators, (1, 2, 4, 8)
                ),
            },
            "action_composition_error": _action_composition_errors(
                bundle.model.operators, bundle.true_operators
            ),
            "planning": {"id": id_plan, "ood": ood_plan},
            "diagnostics": _diagnostics(
                bundle.model,
                bundle.test,
                _make_ground_truth(4, torch.device("cpu"))[0],
            ),
            "protocol": {
                "transition": "s_next = A_group s",
                "rollout_action_schedule": "random_group_sequence",
                "ood_group_one_probability": config.ood_group_one_probability,
                "ood_state_shift": config.ood_state_shift,
                "planning_success_threshold": config.planning_success_threshold,
                "planning_goal": "independent_true_target_sequence",
            },
        }
    if not _finite(record):
        raise FloatingPointError(f"non-finite record: {mode}:{alignment_weight}:{seed}")
    return record


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(
            f"{record['mode']}:{record['alignment_weight']}", []
        ).append(record)

    def mean_std(values: list[float]) -> dict[str, float]:
        mean = sum(values) / len(values)
        std = 0.0 if len(values) < 2 else math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        return {"mean": mean, "std": std}

    output: dict[str, object] = {}
    for key, rows in groups.items():
        def values(path: tuple[str, ...]) -> list[float]:
            output_values: list[float] = []
            for row in rows:
                value: object = row
                for item in path:
                    value = value[item]  # type: ignore[index]
                output_values.append(float(value))
            return output_values

        output[key] = {
            "n_seeds": len(rows),
            "test_one_step_mse": mean_std(values(("test_one_step_mse",))),
            "id_rollout_h8_mse": mean_std(values(("random_rollout_mse", "id", "8"))),
            "ood_rollout_h8_mse": mean_std(values(("random_rollout_mse", "ood", "8"))),
            "composition_error": mean_std(
                [
                    sum(float(v) for v in row["action_composition_error"].values())
                    / len(row["action_composition_error"])
                    for row in rows
                ]
            ),
            "id_planning_regret": mean_std(values(("planning", "id", "planning_regret"))),
            "ood_planning_regret": mean_std(values(("planning", "ood", "planning_regret"))),
            "id_planning_success": mean_std(values(("planning", "id", "goal_success_rate"))),
            "ood_planning_success": mean_std(values(("planning", "ood", "goal_success_rate"))),
            "global_commutator": mean_std(
                values(("diagnostics", "global_commutator"))
            ),
            "conditional_commutator": mean_std(
                values(("diagnostics", "conditional_commutator_mean"))
            ),
        }
    return output


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# Functional consequences of alignment interventions",
        "",
        "This is a CPU-only synthetic action-conditioned benchmark, not a nonlinear LeWM, PushT, online-control, or GPU result.",
        "The alignment modes share the same generated data, optimizer, validation-MSE checkpoint rule, and evaluation code.",
        "",
        "| mode | weight | ID one-step MSE | ID random h=8 MSE | OOD random h=8 MSE | composition error | ID regret | OOD regret | ID success | OOD success | global commutator | conditional commutator |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)

        def fmt(name: str) -> str:
            value = values[name]
            return f"{value['mean']:.6g} +/- {value['std']:.2g}"

        lines.append(
            f"| {mode} | {float(weight):.6g} | {fmt('test_one_step_mse')} | "
            f"{fmt('id_rollout_h8_mse')} | {fmt('ood_rollout_h8_mse')} | "
            f"{fmt('composition_error')} | "
            f"{fmt('id_planning_regret')} | {fmt('ood_planning_regret')} | "
            f"{fmt('id_planning_success')} | {fmt('ood_planning_success')} | "
            f"{fmt('global_commutator')} | "
            f"{fmt('conditional_commutator')} |"
        )
    lines.extend(
        [
            "",
            "The planning columns are controlled candidate-set metrics: an independent true target trajectory defines the goal, the model selects by predicted final-state distance, and physical regret is measured against the oracle candidate set.",
            "",
            "Interpretation boundary: this benchmark tests functional consequences of global versus action-conditional alignment under random action sequences. It does not establish that a lower commutator is causally sufficient for planning, nor does it establish nonlinear LeWM, PushT, online-control, or GPU gains.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    args = parser.parse_args()
    config = FunctionalAlignmentConfig(epochs=args.epochs)
    modes = ["none", "global", "conditional"]
    records: list[dict[str, object]] = []
    start = time.time()
    for mode in modes:
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(
                    f"[functional-alignment] mode={mode} weight={weight} seed={seed}: "
                    "measuring random-sequence rollout and candidate-set planning"
                )
                records.append(run_one(mode, weight, seed, config))
    report = {
        "protocol": {
            **asdict(config),
            "device": "cpu",
            "seeds": args.seeds,
            "modes": modes,
            "weights": args.weights,
            "learned_training": True,
            "synthetic_only": True,
            "random_sequence_rollout": True,
            "candidate_set_planning_proxy": True,
            "elapsed_seconds": time.time() - start,
        },
        "records": records,
        "summary": _summary(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
