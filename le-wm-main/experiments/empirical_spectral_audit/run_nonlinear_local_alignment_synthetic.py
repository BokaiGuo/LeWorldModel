"""Functional ablation for local alignment in a nonlinear action world.

The earlier learned alignment benchmark uses action-conditioned linear branches.
This follow-up keeps the experiment CPU-sized but makes the transition
nonlinear and state dependent.  The ground-truth world has an involutive
state/action symmetry, so predictor equivariance is a meaningful comparator;
the alignment interventions instead penalize covariance--local-Jacobian
non-commutativity.  The experiment is a synthetic functional proxy, not a
nonlinear LeWM or PushT result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NonlinearLocalConfig:
    state_dim: int = 4
    hidden_dim: int = 64
    train_samples: int = 512
    validation_samples: int = 256
    test_samples: int = 1024
    epochs: int = 160
    learning_rate: float = 2e-3
    noise_std: float = 0.02
    local_anchors: int = 32
    local_eps: float = 2e-3
    local_min_samples: int = 16
    rollout_trajectories: int = 256
    rollout_horizon: int = 8
    planning_tasks: int = 64
    planning_candidates: int = 48
    planning_horizon: int = 4
    planning_success_threshold: float = 0.25
    ood_state_shift: float = 1.5
    alignment_weight: float = 0.02
    predictor_weight: float = 1.0


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _symmetry_matrix(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # An involution: R @ R = I.  This makes action swap + state transform a
    # genuine order-two symmetry of the constructed transition.
    return torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device, dtype=dtype))


def _transform_state(state: torch.Tensor) -> torch.Tensor:
    return state @ _symmetry_matrix(state.device, state.dtype).T


def _base_transition(state: torch.Tensor) -> torch.Tensor:
    matrix = torch.tensor(
        [[0.70, 0.10, 0.00, 0.00], [0.00, 0.64, 0.08, 0.00],
         [0.00, 0.00, 0.58, 0.06], [0.04, 0.00, 0.00, 0.50]],
        device=state.device,
        dtype=state.dtype,
    )
    linear = state @ matrix.T
    nonlinear = torch.stack(
        [
            0.16 * torch.tanh(state[:, 0] * state[:, 1]),
            0.10 * torch.tanh(state[:, 2] - state[:, 3]),
            0.12 * torch.tanh(state[:, 0]),
            0.08 * torch.tanh(state[:, 1] + state[:, 3]),
        ],
        dim=-1,
    )
    return linear + nonlinear


def true_transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Nonlinear stable transition with an exact action/state symmetry."""

    labels = action.reshape(-1).long().clamp(0, 1)
    base = _base_transition(state)
    transformed_state = _transform_state(state)
    transformed_base = _base_transition(transformed_state)
    branch_one = _transform_state(transformed_base)
    return torch.where(labels[:, None].bool(), branch_one, base)


class NonlinearWorldModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        labels = action.reshape(-1).long().clamp(0, 1)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=2).to(state.dtype)
        return self.network(torch.cat((state.float(), one_hot), dim=-1))


def _sample_split(
    count: int,
    generator: torch.Generator,
    noise_std: float,
    *,
    group_one_probability: float = 0.5,
    state_shift: float = 0.0,
) -> dict[str, torch.Tensor]:
    state = torch.randn(count, 4, generator=generator) + state_shift
    action = (torch.rand(count, generator=generator) < group_one_probability).long()
    target = true_transition(state, action)
    if noise_std > 0:
        target = target + noise_std * torch.randn(target.shape, generator=generator)
    return {"state": state, "action": action, "target": target}


def _subsample(split: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    if count >= split["state"].size(0):
        return split
    indices = torch.arange(split["state"].size(0))[:count]
    return {key: value[indices] for key, value in split.items()}


def _secant_jacobian(
    model: NonlinearWorldModel,
    state: torch.Tensor,
    action: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Finite-difference local Jacobians with gradients through model outputs."""

    base = model(state, action)
    columns: list[torch.Tensor] = []
    for coordinate in range(state.size(-1)):
        perturbation = torch.zeros_like(state)
        perturbation[:, coordinate] = float(eps)
        columns.append((model(state + perturbation, action) - base) / float(eps))
    return torch.stack(columns, dim=-1)


def _normalized_commutator(covariance: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    commutator = covariance @ operator - operator @ covariance
    denominator = covariance.square().sum() * operator.square().sum() + 1e-8
    return commutator.square().sum() / denominator


def _covariance(state: torch.Tensor) -> torch.Tensor:
    centered = state - state.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(state.size(0) - 1, 1)


def _local_alignment_penalty(
    state: torch.Tensor,
    action: torch.Tensor,
    jacobians: torch.Tensor,
    mode: str,
    min_samples: int,
    reference_state: torch.Tensor | None = None,
    reference_action: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode not in {"global", "conditional"}:
        return jacobians.new_zeros(())
    reference_state = state if reference_state is None else reference_state
    reference_action = action if reference_action is None else reference_action
    if mode == "global":
        covariance = _covariance(reference_state).detach()
        return _normalized_commutator(covariance, jacobians.mean(dim=0))

    # State-local geometry: for each anchor, fit the covariance of its nearest
    # same-action reference states.  The covariance is detached, while the
    # local Jacobian remains differentiable through the predictor outputs.
    reference_state_detached = reference_state.detach()
    reference_action_detached = reference_action.reshape(-1).long().detach()
    anchor_actions = action.reshape(-1).long().detach()
    penalties: list[torch.Tensor] = []
    for index in range(state.size(0)):
        distances = (reference_state_detached - state[index].detach()).square().sum(dim=-1)
        distances = distances.masked_fill(reference_action_detached != anchor_actions[index], float("inf"))
        available = int(torch.isfinite(distances).sum())
        if available < int(min_samples):
            continue
        neighbours = torch.topk(distances, k=min(int(min_samples), available), largest=False).indices
        covariance = _covariance(reference_state_detached[neighbours]).detach()
        penalties.append(_normalized_commutator(covariance, jacobians[index]))
    return torch.stack(penalties).mean() if penalties else jacobians.new_zeros(())


def _predictor_equivariance_loss(
    model: NonlinearWorldModel, state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    prediction = model(state, action)
    transformed_action = 1 - action
    transformed_prediction = model(_transform_state(state), transformed_action)
    expected = _transform_state(prediction)
    return (transformed_prediction - expected).square().mean()


def _random_jacobian_loss(jacobians: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Direction-randomized Jacobian regularizer used by the control arm."""

    return (jacobians - target.to(device=jacobians.device, dtype=jacobians.dtype)).square().mean()


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = [gradient.square().sum() for gradient in gradients if gradient is not None]
    return torch.stack(squared).sum().sqrt() if squared else loss.new_zeros(())


@torch.no_grad()
def _rollout_mse(
    model: NonlinearWorldModel,
    count: int,
    generator: torch.Generator,
    horizon: int,
    state_shift: float,
    group_one_probability: float,
) -> dict[str, float]:
    state = torch.randn(count, 4, generator=generator) + state_shift
    actions = (torch.rand(count, horizon, generator=generator) < group_one_probability).long()
    predicted = state.clone()
    truth = state.clone()
    output: dict[str, float] = {}
    for step in range(horizon):
        action = actions[:, step]
        predicted = model(predicted, action)
        truth = true_transition(truth, action)
        current_horizon = step + 1
        if current_horizon in {1, 2, 4, 8}:
            output[str(current_horizon)] = float((predicted - truth).square().mean())
    return output


@torch.no_grad()
def _composition_error(
    model: NonlinearWorldModel,
    count: int,
    generator: torch.Generator,
    state_shift: float,
) -> float:
    state = torch.randn(count, 4, generator=generator) + state_shift
    first = (torch.rand(count, generator=generator) < 0.5).long()
    second = (torch.rand(count, generator=generator) < 0.5).long()
    predicted = model(model(state, first), second)
    truth = true_transition(true_transition(state, first), second)
    return float((predicted - truth).square().mean().sqrt())


@torch.no_grad()
def _planning_proxy(
    model: NonlinearWorldModel,
    count: int,
    candidates: int,
    horizon: int,
    generator: torch.Generator,
    state_shift: float,
) -> dict[str, float]:
    initial = torch.randn(count, 4, generator=generator) + state_shift
    actions = (torch.rand(count, candidates, horizon, generator=generator) < 0.5).long()
    flat_initial = initial[:, None, :].expand(-1, candidates, -1).reshape(-1, 4)
    predicted = flat_initial
    truth = flat_initial
    flat_actions = actions.reshape(-1, horizon)
    for step in range(horizon):
        action = flat_actions[:, step]
        predicted = model(predicted, action)
        truth = true_transition(truth, action)
    predicted = predicted.reshape(count, candidates, -1)
    truth = truth.reshape(count, candidates, -1)
    goal = truth[:, 0]
    predicted_distances = (predicted - goal[:, None, :]).square().sum(dim=-1)
    chosen = predicted_distances.argmin(dim=1)
    chosen_truth = truth[torch.arange(count), chosen]
    oracle_distances = (truth - goal[:, None, :]).square().sum(dim=-1)
    oracle = oracle_distances.min(dim=1).values
    selected = (chosen_truth - goal).square().sum(dim=-1).sqrt()
    oracle_distance = oracle.sqrt()
    regret = selected.mean() - oracle_distance.mean()
    return {
        "selected_goal_distance": float(selected.mean()),
        "oracle_goal_distance": float(oracle_distance.mean()),
        "planning_regret": float(regret),
        "goal_success_rate": float((selected <= 0.25).float().mean()),
        "oracle_success_rate": float((oracle_distance <= 0.25).float().mean()),
        "oracle_choice_agreement": float((chosen == oracle_distances.argmin(dim=1)).float().mean()),
    }


@torch.no_grad()
def _diagnostics(
    model: NonlinearWorldModel,
    split: dict[str, torch.Tensor],
    config: NonlinearLocalConfig,
) -> dict[str, float]:
    anchors = _subsample(split, min(config.test_samples, 256))
    jacobians = _secant_jacobian(model, anchors["state"], anchors["action"], config.local_eps)
    global_value = float(
        _local_alignment_penalty(
            anchors["state"], anchors["action"], jacobians, "global", config.local_min_samples,
            reference_state=split["state"], reference_action=split["action"],
        )
    )
    conditional_value = float(
        _local_alignment_penalty(
            anchors["state"], anchors["action"], jacobians, "conditional", config.local_min_samples,
            reference_state=split["state"], reference_action=split["action"],
        )
    )
    transformed_state = _transform_state(anchors["state"])
    transformed_action = 1 - anchors["action"]
    prediction = model(anchors["state"], anchors["action"])
    transformed_prediction = model(transformed_state, transformed_action)
    equivariance = float((transformed_prediction - _transform_state(prediction)).square().mean().sqrt())
    return {
        "global_local_commutator": global_value,
        "conditional_local_commutator": conditional_value,
        "predictor_equivariance_error": equivariance,
    }


def _fit_model(
    mode: str,
    seed: int,
    config: NonlinearLocalConfig,
) -> tuple[NonlinearWorldModel, dict[str, torch.Tensor], dict[str, torch.Tensor], int, float, float]:
    if mode not in {"none", "global", "conditional", "predictor", "both", "random", "anti"}:
        raise ValueError(f"unknown mode: {mode}")
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = _sample_split(config.train_samples, generator, config.noise_std)
    validation = _sample_split(config.validation_samples, generator, config.noise_std)
    test = _sample_split(config.test_samples, generator, 0.0)
    model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    random_scale_sum = 0.0
    random_target_generator = torch.Generator().manual_seed(seed + 20_000)
    random_target = 0.5 * torch.randn(config.state_dim, config.state_dim, generator=random_target_generator)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        anchors = _subsample(train, config.local_anchors)
        prediction = model(train["state"], train["action"])
        prediction_loss = (prediction - train["target"]).square().mean()
        jacobians = _secant_jacobian(model, anchors["state"], anchors["action"], config.local_eps)
        alignment_mode = "conditional" if mode in {"conditional", "both", "anti"} else "global" if mode == "global" else "none"
        alignment_loss = _local_alignment_penalty(
            anchors["state"], anchors["action"], jacobians, alignment_mode, config.local_min_samples,
            reference_state=train["state"], reference_action=train["action"],
        )
        predictor_loss = _predictor_equivariance_loss(model, anchors["state"], anchors["action"])
        predictor_weight = config.predictor_weight if mode in {"predictor", "both"} else 0.0
        if mode == "random":
            random_loss = _random_jacobian_loss(jacobians, random_target)
            alignment_reference = _local_alignment_penalty(
                anchors["state"], anchors["action"], jacobians, "conditional", config.local_min_samples,
                reference_state=train["state"], reference_action=train["action"],
            )
            alignment_gradient_norm = _gradient_norm(alignment_reference, parameters)
            random_gradient_norm = _gradient_norm(random_loss, parameters)
            random_scale = (alignment_gradient_norm / random_gradient_norm.clamp_min(1e-12)).detach()
            random_scale_sum += float(random_scale)
            objective = prediction_loss + random_scale * random_loss
        else:
            signed_alignment_weight = -config.alignment_weight if mode == "anti" else config.alignment_weight
            objective = prediction_loss + signed_alignment_weight * alignment_loss + predictor_weight * predictor_loss
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            validation_prediction = model(validation["state"], validation["action"])
            validation_mse = float((validation_prediction - validation["target"]).square().mean())
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    return model, train, test, best_epoch, best_validation, random_scale_sum / max(config.epochs, 1)


def run_one(mode: str, seed: int, config: NonlinearLocalConfig) -> dict[str, Any]:
    model, train, test, best_epoch, best_validation, random_scale_mean = _fit_model(mode, seed, config)
    eval_generator = torch.Generator().manual_seed(seed + 50_000)
    ood_generator = torch.Generator().manual_seed(seed + 60_000)
    with torch.no_grad():
        prediction = model(test["state"], test["action"])
        test_mse = float((prediction - test["target"]).square().mean())
    record: dict[str, Any] = {
        "mode": mode,
        "alignment_weight": config.alignment_weight if mode in {"global", "conditional", "both", "anti"} else 0.0,
        "predictor_weight": config.predictor_weight if mode in {"predictor", "both"} else 0.0,
        "random_control_gradient_scale_mean": random_scale_mean,
        "seed": seed,
        "device": "cpu",
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "test_one_step_mse": test_mse,
        "id_rollout_mse": _rollout_mse(model, config.rollout_trajectories, eval_generator, config.rollout_horizon, 0.0, 0.5),
        "ood_rollout_mse": {
            "state_shift": _rollout_mse(model, config.rollout_trajectories, ood_generator, config.rollout_horizon, config.ood_state_shift, 0.5),
            "action_imbalance": _rollout_mse(model, config.rollout_trajectories, ood_generator, config.rollout_horizon, 0.0, 0.85),
        },
        "composition_error": {
            "id": _composition_error(model, config.rollout_trajectories, eval_generator, 0.0),
            "state_shift": _composition_error(model, config.rollout_trajectories, ood_generator, config.ood_state_shift),
        },
        "planning": {
            "id": _planning_proxy(model, config.planning_tasks, config.planning_candidates, config.planning_horizon, eval_generator, 0.0),
            "state_shift": _planning_proxy(model, config.planning_tasks, config.planning_candidates, config.planning_horizon, ood_generator, config.ood_state_shift),
        },
        "diagnostics": _diagnostics(model, test, config),
        "protocol": {
            "transition": "nonlinear state-dependent action-symmetric map",
            "local_operator": "finite-difference Jacobian of predictor",
            "local_cells": "nearest same-action state neighbourhoods",
            "ood_state_shift": config.ood_state_shift,
            "planning_goal": "first true candidate endpoint",
        },
    }
    _assert_finite(record)
    return record


def _assert_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in sorted({str(row["mode"]) for row in records}):
        rows = [row for row in records if row["mode"] == mode]
        def mean(key: str) -> float:
            return float(sum(float(row[key]) for row in rows) / len(rows))
        output[mode] = {
            "n_seeds": len(rows),
            "test_one_step_mse_mean": mean("test_one_step_mse"),
            "id_rollout_h8_mse_mean": float(sum(row["id_rollout_mse"]["8"] for row in rows) / len(rows)),
            "ood_state_shift_rollout_h8_mse_mean": float(sum(row["ood_rollout_mse"]["state_shift"]["8"] for row in rows) / len(rows)),
            "ood_state_shift_planning_regret_mean": float(sum(row["planning"]["state_shift"]["planning_regret"] for row in rows) / len(rows)),
            "global_local_commutator_mean": float(sum(row["diagnostics"]["global_local_commutator"] for row in rows) / len(rows)),
            "conditional_local_commutator_mean": float(sum(row["diagnostics"]["conditional_local_commutator"] for row in rows) / len(rows)),
            "predictor_equivariance_error_mean": float(sum(row["diagnostics"]["predictor_equivariance_error"] for row in rows) / len(rows)),
        }
    return output


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear local-alignment functional benchmark",
        "",
        "This CPU-only benchmark uses a nonlinear state-dependent transition with an exact action/state involution. Local alignment is computed from finite-difference predictor Jacobians inside action-by-state cells. It is synthetic mechanism evidence, not nonlinear LeWM, PushT, online-control, or causal-control evidence.",
        "",
        "## Protocol",
        "",
        f"- seeds: `{report['protocol']['seeds']}`; epochs: `{report['protocol']['epochs']}`; train/validation/test: `{report['protocol']['train_samples']}/{report['protocol']['validation_samples']}/{report['protocol']['test_samples']}`",
        f"- conditions: `{report['protocol']['conditions']}`; alignment weight: `{report['protocol']['alignment_weight']}`; predictor weight: `{report['protocol']['predictor_weight']}`",
        "- all conditions share generated splits, model class, optimizer, validation-MSE checkpoint selection, rollouts, composition, and planning proxy",
        "- planning selects among candidate action sequences using predicted endpoints and scores the selected sequence using the true transition",
        "",
        "| condition | n | one-step MSE | ID rollout h=8 | OOD state-shift rollout h=8 | OOD planning regret | global local C | conditional local C | predictor equivariance error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in report["summary"].items():
        lines.append(
            f"| {mode} | {values['n_seeds']} | {values['test_one_step_mse_mean']:.6g} | {values['id_rollout_h8_mse_mean']:.6g} | {values['ood_state_shift_rollout_h8_mse_mean']:.6g} | {values['ood_state_shift_planning_regret_mean']:.6g} | {values['global_local_commutator_mean']:.6g} | {values['conditional_local_commutator_mean']:.6g} | {values['predictor_equivariance_error_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Conditional local alignment is a functional intervention in this controlled nonlinear proxy, not evidence that lower commutator is universally beneficial.",
            "- The planning metric is a candidate-set proxy and is not online task success or causal control.",
            "- Matched one-step-error inference is reported separately; matching filters cannot equalize all optimization or representation differences.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--local-anchors", type=int, default=128)
    parser.add_argument("--alignment-weight", type=float, default=0.02)
    parser.add_argument("--predictor-weight", type=float, default=1.0)
    parser.add_argument("--include-controls", action="store_true")
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
    records: list[dict[str, Any]] = []
    started = time.time()
    conditions = ["none", "global", "conditional", "predictor", "both"]
    if args.include_controls:
        conditions.extend(["random", "anti"])
    print("Running nonlinear local functional ablation: compare global/local Jacobian alignment and predictor equivariance under shared nonlinear dynamics.")
    for seed in args.seeds:
        for mode in conditions:
            record = run_one(mode, int(seed), config)
            records.append(record)
            print(f"seed={seed} mode={mode} one_step={record['test_one_step_mse']:.6g} ood_h8={record['ood_rollout_mse']['state_shift']['8']:.6g}")
    report = {
        "protocol": {
            **asdict(config),
            "seeds": [int(seed) for seed in args.seeds],
            "conditions": list(conditions),
            "learned_nonlinear_predictor": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": _summary(records),
    }
    _assert_finite(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear local-alignment functional ablation\n\n"
        "This directory contains the shared-protocol nonlinear synthetic benchmark. "
        "The JSON stores all per-seed records; the Markdown stores aggregate metrics and claim boundaries.\n\n"
        f"- source: `experiments/empirical_spectral_audit/run_nonlinear_local_alignment_synthetic.py`\n"
        f"- records: `{args.output.name}`\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
