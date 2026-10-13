"""Learned synthetic benchmark for the action-conditioned alignment intervention.

The benchmark is deliberately small but is a training experiment rather than a
fixed-operator mechanism smoke.  Two action groups have different covariance
geometries and different transition matrices.  Within each group, the
transition commutes with the local covariance; after pooling the groups, a
single global covariance/operator pair generally does not commute.  A
two-branch linear predictor is trained under the same protocol with no
alignment, global alignment, or conditional alignment.

The artifact is intended to answer a narrow causal question: does a local
alignment intervention preserve or improve held-out prediction and rollout
when the data-generating process has action-conditional geometry?  It is not a
claim about the nonlinear LeWM model or PushT performance.

The reported rollout uses the sample's action label repeatedly at every step.
It is therefore a repeated-action stability diagnostic, not an evaluation over
arbitrary action sequences. Ordered two-action composition is measured
separately from the learned operator products.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
    )
except ModuleNotFoundError:  # direct ``python experiments/.../script.py`` invocation
    from regularizers import operator_alignment_diagnostic, operator_alignment_loss


@dataclass(frozen=True)
class BenchmarkConfig:
    dim: int = 4
    train_samples_per_group: int = 512
    validation_samples_per_group: int = 512
    test_samples_per_group: int = 2048
    epochs: int = 300
    learning_rate: float = 0.05
    ridge: float = 1e-3
    min_samples: int = 32
    alignment_bins: int = 2
    noise_std: float = 0.02
    rollout_horizons: tuple[int, ...] = (1, 2, 4, 8)


def _rotation(dim: int, angle_degrees: float, device: torch.device) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.eye(dim, device=device)
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    return rotation


def _make_ground_truth(dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # The groups intentionally have different covariance and transition
    # spectra.  Using the same spectra under a rotation can make the pooled
    # covariance and pooled operator accidentally commute, weakening the test
    # of global-vs-conditional alignment.
    group_scales = (
        torch.tensor([10.0, 0.10, 0.50, 0.05], device=device)[:dim],
        torch.tensor([0.20, 8.00, 1.50, 0.10], device=device)[:dim],
    )
    group_eigenvalues = (
        torch.tensor([0.95, 0.10, 0.60, 0.20], device=device)[:dim],
        torch.tensor([0.80, 0.15, 0.30, 0.05], device=device)[:dim],
    )
    operators = []
    covariances = []
    for group, angle in enumerate((0.0, 47.0)):
        rotation = _rotation(dim, angle, device)
        covariance = rotation @ torch.diag(group_scales[group]) @ rotation.T
        operator = rotation @ torch.diag(group_eigenvalues[group]) @ rotation.T
        covariances.append(covariance)
        operators.append(operator)
    return torch.stack(covariances), torch.stack(operators)


def _population_geometry(covariances: torch.Tensor, operators: torch.Tensor) -> dict[str, object]:
    """Return exact local and pooled geometry checks for the synthetic process."""

    pooled_covariance = covariances.mean(dim=0)
    # For y = A_g x and row samples, E[x y^T] = Sigma_g A_g^T.
    pooled_cross = torch.stack([covariances[g] @ operators[g].T for g in range(2)]).mean(dim=0)
    pooled_operator = torch.linalg.solve(pooled_covariance, pooled_cross).T
    return {
        "local_commutator_by_group": [_commutator(covariances[g], operators[g]) for g in range(2)],
        "pooled_commutator": _commutator(pooled_covariance, pooled_operator),
    }


def _sample_split(
    samples_per_group: int,
    covariances: torch.Tensor,
    operators: torch.Tensor,
    noise_std: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    dim = covariances.size(-1)
    states = []
    next_states = []
    actions = []
    for group in range(2):
        chol = torch.linalg.cholesky(covariances[group])
        source = torch.randn(samples_per_group, dim, generator=generator, device=covariances.device) @ chol.T
        target = source @ operators[group].T
        if noise_std > 0:
            target = target + noise_std * torch.randn(
                target.shape, generator=generator, device=target.device
            )
        states.append(source)
        next_states.append(target)
        actions.append(torch.full((samples_per_group, 1), float(group), device=source.device))
    return {
        "source": torch.cat(states),
        "target": torch.cat(next_states),
        "action": torch.cat(actions),
    }


class ActionConditionedLinear(torch.nn.Module):
    """Two-branch linear predictor; only the branch selection is conditioned on action."""

    def __init__(self, dim: int, generator: torch.Generator, device: torch.device) -> None:
        super().__init__()
        initial = 0.15 * torch.randn(2, dim, dim, generator=generator, device=device)
        self.operators = torch.nn.Parameter(initial)

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        labels = action.reshape(-1).long().clamp(0, 1)
        selected = self.operators[labels]
        return torch.bmm(selected, source.unsqueeze(-1)).squeeze(-1)


def _predict(model: ActionConditionedLinear, split: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(split["source"], split["action"])


def _mse_by_group(error: torch.Tensor, actions: torch.Tensor) -> dict[str, float]:
    labels = actions.reshape(-1).long()
    return {str(group): float(error[labels == group].square().mean().detach().cpu()) for group in (0, 1)}


@torch.no_grad()
def _rollout_mse(
    model: ActionConditionedLinear,
    split: dict[str, torch.Tensor],
    horizons: tuple[int, ...],
    true_operators: torch.Tensor,
) -> dict[str, float]:
    # Each sample keeps its action label fixed across the rollout. This makes
    # the horizon metric a repeated-action stability test; arbitrary action
    # sequences are covered only by the separate composition diagnostic.
    source = split["source"]
    labels = split["action"].reshape(-1).long()
    current = source
    truth = source
    result: dict[str, float] = {}
    for horizon in range(1, max(horizons) + 1):
        current = model(current, split["action"])
        truth = torch.bmm(true_operators[labels], truth.unsqueeze(-1)).squeeze(-1)
        if horizon in horizons:
            result[str(horizon)] = float((current - truth).square().mean().detach().cpu())
    return result


def _commutator(covariance: torch.Tensor, operator: torch.Tensor) -> float:
    comm = covariance @ operator - operator @ covariance
    denom = covariance.square().sum() * operator.square().sum() + 1e-12
    return float((comm.square().sum() / denom).detach().cpu())


@torch.no_grad()
def _action_composition_errors(
    learned_operators: torch.Tensor,
    true_operators: torch.Tensor,
) -> dict[str, float]:
    """Measure ordered two-action composition error.

    Operators act on column latents.  For an action pair ``(first, second)``,
    the composed map is therefore ``A_second @ A_first``.  The metric compares
    the learned and data-generating composed maps, normalized by the norm of
    the true composition.  It is a synthetic action-algebra diagnostic, not a
    claim that a nonlinear LeWM predictor is globally linear.
    """

    errors: dict[str, float] = {}
    for first in range(true_operators.size(0)):
        for second in range(true_operators.size(0)):
            learned = learned_operators[second] @ learned_operators[first]
            truth = true_operators[second] @ true_operators[first]
            denominator = truth.square().sum().sqrt().clamp_min(1e-12)
            errors[f"{first}->{second}"] = float(
                ((learned - truth).square().sum().sqrt() / denominator).cpu()
            )
    return errors


@torch.no_grad()
def _diagnostics(
    model: ActionConditionedLinear,
    split: dict[str, torch.Tensor],
    true_covariances: torch.Tensor,
) -> dict[str, object]:
    prediction = _predict(model, split)
    source = split["source"]
    labels = split["action"].reshape(-1).long()
    source_centered = source - source.mean(dim=0, keepdim=True)
    covariance = source_centered.T @ source_centered / max(source.size(0) - 1, 1)
    fitted = []
    conditional_values = []
    for group in (0, 1):
        mask = labels == group
        group_source = source[mask]
        group_prediction = prediction[mask]
        centered = group_source - group_source.mean(dim=0, keepdim=True)
        group_cov = centered.T @ centered / max(group_source.size(0) - 1, 1)
        cross = centered.T @ (group_prediction - group_prediction.mean(dim=0, keepdim=True))
        fitted_operator = torch.linalg.solve(group_cov + 1e-3 * torch.eye(source.size(1), device=source.device), cross).T
        fitted.append(fitted_operator)
        conditional_values.append(_commutator(group_cov, fitted_operator))
    pooled_cross = source_centered.T @ (prediction - prediction.mean(dim=0, keepdim=True))
    pooled_operator = torch.linalg.solve(covariance + 1e-3 * torch.eye(source.size(1), device=source.device), pooled_cross).T
    diagnostic = operator_alignment_diagnostic(
        source, prediction, actions=split["action"], mode="conditional", num_bins=2, min_samples=32
    )
    return {
        "global_commutator": _commutator(covariance, pooled_operator),
        "conditional_commutator_mean": float(sum(conditional_values) / len(conditional_values)),
        "conditional_commutator_by_group": conditional_values,
        "regularizer_conditional_diagnostic": diagnostic,
        "true_local_commutator_max": max(_commutator(true_covariances[g], fitted[g]) for g in (0, 1)),
    }


def run_one(
    mode: str,
    alignment_weight: float,
    seed: int,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, object]:
    data_generator = torch.Generator(device=device).manual_seed(seed)
    model_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    true_covariances, true_operators = _make_ground_truth(config.dim, device)
    train = _sample_split(config.train_samples_per_group, true_covariances, true_operators, config.noise_std, data_generator)
    validation = _sample_split(config.validation_samples_per_group, true_covariances, true_operators, config.noise_std, data_generator)
    test = _sample_split(config.test_samples_per_group, true_covariances, true_operators, 0.0, data_generator)
    model = ActionConditionedLinear(config.dim, model_generator, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, train)
        prediction_mse = (prediction - train["target"]).square().mean()
        alignment = operator_alignment_loss(
            train["source"],
            prediction,
            actions=train["action"],
            mode=mode,
            num_bins=config.alignment_bins,
            min_samples=config.min_samples,
            ridge=config.ridge,
        )
        objective = prediction_mse + float(alignment_weight) * alignment
        objective.backward()
        optimizer.step()
        if epoch in (1, config.epochs):
            with torch.no_grad():
                validation_mse = (_predict(model, validation) - validation["target"]).square().mean()
            history.append({
                "epoch": float(epoch),
                "train_prediction_mse": float(prediction_mse.detach().cpu()),
                "alignment_loss": float(alignment.detach().cpu()),
                "validation_mse": float(validation_mse.detach().cpu()),
            })
    with torch.no_grad():
        test_prediction = _predict(model, test)
        test_error = test_prediction - test["target"]
        test_mse = float(test_error.square().mean().detach().cpu())
        composition_errors = _action_composition_errors(model.operators, true_operators)
    return {
        "mode": mode,
        "alignment_weight": alignment_weight,
        "seed": seed,
        "device": str(device),
        "test_one_step_mse": test_mse,
        "test_one_step_mse_by_group": _mse_by_group(test_error, test["action"]),
        "rollout_mse": _rollout_mse(model, test, config.rollout_horizons, true_operators),
        "action_composition_error": composition_errors,
        "diagnostics": _diagnostics(model, test, true_covariances),
        "learned_operators": model.operators.detach().cpu().tolist(),
        "history": history,
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        key = f"{record['mode']}:{record['alignment_weight']}"
        grouped.setdefault(key, []).append(record)

    output: dict[str, object] = {}
    for key, rows in grouped.items():
        one_step = [float(row["test_one_step_mse"]) for row in rows]
        rollout = {
            horizon: [float(row["rollout_mse"][horizon]) for row in rows]  # type: ignore[index]
            for horizon in rows[0]["rollout_mse"]  # type: ignore[union-attr]
        }
        global_comm = [float(row["diagnostics"]["global_commutator"]) for row in rows]  # type: ignore[index]
        conditional_comm = [float(row["diagnostics"]["conditional_commutator_mean"]) for row in rows]  # type: ignore[index]
        composition_keys = sorted(rows[0]["action_composition_error"])  # type: ignore[index]
        composition_mean = {
            key: sum(float(row["action_composition_error"][key]) for row in rows) / len(rows)  # type: ignore[index]
            for key in composition_keys
        }
        output[key] = {
            "n_seeds": len(rows),
            "test_one_step_mse_mean": sum(one_step) / len(one_step),
            "test_one_step_mse_std": float(torch.tensor(one_step).std(unbiased=False)),
            "rollout_mse_mean": {h: sum(values) / len(values) for h, values in rollout.items()},
            "global_commutator_mean": sum(global_comm) / len(global_comm),
            "conditional_commutator_mean": sum(conditional_comm) / len(conditional_comm),
            "action_composition_error_mean": composition_mean,
            "action_composition_error_overall_mean": sum(composition_mean.values()) / len(composition_mean),
        }
    return output


def _write_markdown(report: dict[str, object], output: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Learned action-conditioned alignment benchmark",
        "",
        "This is a synthetic learned-predictor experiment, not a learned LeWM or PushT result.",
        "The data-generating transition commutes with each action-local covariance, while pooled geometry is intentionally heterogeneous.",
        "",
        "## Protocol",
        "",
        f"- device: `{report['protocol']['device']}`; seeds: `{report['protocol']['seeds']}`; epochs: `{report['protocol']['epochs']}`",
        f"- modes: `{report['protocol']['modes']}`; weights: `{report['protocol']['weights']}`",
        f"- exact data-generating local commutators: `{report['protocol']['ground_truth_geometry']['local_commutator_by_group']}`; pooled commutator: `{report['protocol']['ground_truth_geometry']['pooled_commutator']:.6g}`",
        "- all modes use identical generated splits, initialization seeds, optimizer, and evaluation code",
        "- reported outcomes include one-step prediction, group-wise error, multi-step rollout, and global/conditional commutators",
        "- rollout uses the same action label repeatedly at every step; it is not an arbitrary action-sequence evaluation",
        "- action composition error compares each learned two-action operator product with the corresponding data-generating product",
        "",
        "## Aggregate results",
        "",
        "| intervention | weight | seeds | test MSE | rollout h=8 MSE | composition error | global commutator | conditional commutator |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in summary.items():
        mode, weight = key.split(":", 1)
        rollout = values["rollout_mse_mean"]
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | "
            f"{values['test_one_step_mse_mean']:.6g} | {rollout['8']:.6g} | "
            f"{values['action_composition_error_overall_mean']:.6g} | "
            f"{values['global_commutator_mean']:.6g} | {values['conditional_commutator_mean']:.6g} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A lower commutator is not treated as sufficient evidence of better prediction or planning. The intended comparison is whether conditional alignment preserves the local geometry and prediction/rollout, while global alignment exposes the cost of imposing one pooled geometry.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    config = BenchmarkConfig(epochs=args.epochs)
    modes = ["none", "global", "conditional"]
    records = []
    start = time.time()
    for mode in modes:
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(f"[benchmark] mode={mode} weight={weight} seed={seed}: measuring prediction/rollout consequences")
                records.append(run_one(mode, weight, seed, config, device))
    report = {
        "protocol": {
            **asdict(config),
            "rollout_horizons": list(config.rollout_horizons),
            "device": str(device),
            "seeds": args.seeds,
            "modes": modes,
            "weights": args.weights,
            "composition_pairs": ["0->0", "0->1", "1->0", "1->1"],
            "rollout_action_schedule": "repeat_same_action",
            "learned_training": True,
            "synthetic_only": True,
            "elapsed_seconds": time.time() - start,
            "ground_truth_geometry": _population_geometry(
                *_make_ground_truth(config.dim, device)
            ),
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
