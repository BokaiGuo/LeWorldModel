"""Train a controlled action-conditioned world model with equivariance ablations.

This benchmark isolates the distinction between latent equivariance and
predictor equivariance.  States and actions are four-dimensional vectors made
of two-dimensional rotation blocks, and the ground-truth transition is
``s_next = s + a``.  A 90-degree block rotation is therefore an exact symmetry
of the data-generating dynamics.  The encoder remains trainable, so the
experiment tests the proposed losses in a learned representation rather than
in a fixed hand-designed feature space.

The artifact is a mechanism-level synthetic benchmark.  It is not a
LeWorldModel, PushT, planning, or GPU result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

try:
    from experiments.empirical_spectral_audit.regularizers import (
        equivariance_diagnostics,
        latent_equivariance_loss,
        predictor_equivariance_loss,
    )
except ModuleNotFoundError:  # direct invocation from the standalone package
    sys.path.append(str(Path(__file__).resolve().parent))
    from regularizers import (
        equivariance_diagnostics,
        latent_equivariance_loss,
        predictor_equivariance_loss,
    )


@dataclass(frozen=True)
class EquivarianceBenchmarkConfig:
    state_dim: int = 4
    latent_dim: int = 8
    train_samples: int = 512
    validation_samples: int = 256
    test_samples: int = 512
    rollout_trajectories: int = 256
    rollout_horizon: int = 8
    epochs: int = 120
    early_stopping_patience: int = 15
    learning_rate: float = 2e-3


@dataclass
class TrainedBenchmark:
    model: "EquivarianceWorldModel"
    train: dict[str, torch.Tensor]
    validation: dict[str, torch.Tensor]
    test: dict[str, torch.Tensor]
    rollout: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_mse: float
    history: list[dict[str, float]]


def _block_rotation(angle: float, blocks: int) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    block = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    return torch.block_diag(*[block for _ in range(blocks)])


ACTION_TRANSFORM = _block_rotation(math.pi / 2.0, blocks=2)
LATENT_TRANSFORM = _block_rotation(math.pi / 2.0, blocks=4)


class EquivarianceWorldModel(nn.Module):
    """Small trainable encoder/predictor pair for the controlled vector world."""

    def __init__(self, state_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + state_dim, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state.float())

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat((latent.float(), action.float()), dim=-1))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _transform(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return values @ matrix.to(device=values.device, dtype=values.dtype).T


def _make_one_step_split(
    count: int, config: EquivarianceBenchmarkConfig, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    state = torch.randn(count, config.state_dim, generator=generator)
    action = 0.25 * torch.randn(count, config.state_dim, generator=generator)
    target = state + action
    return {"state": state, "action": action, "target": target}


def _make_rollout_split(
    count: int, config: EquivarianceBenchmarkConfig, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    base = torch.randn(count, config.state_dim, generator=generator)
    actions = 0.25 * torch.randn(
        count, config.rollout_horizon, config.state_dim, generator=generator
    )
    cumulative = torch.zeros_like(base)
    states = [base]
    for step in range(config.rollout_horizon):
        cumulative = cumulative + actions[:, step]
        states.append(base + cumulative)
    return {"states": torch.stack(states, dim=1), "actions": actions}


def _stable_rank(latent: torch.Tensor) -> float:
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(latent.size(0) - 1, 1)
    trace = covariance.trace()
    return float((trace.square() / covariance.square().sum().clamp_min(1e-12)).detach())


def _validate_protocol(config: EquivarianceBenchmarkConfig) -> None:
    """Fail early if the stated row-vector transform convention is broken."""

    if config.state_dim != 4 or config.latent_dim != 8:
        raise ValueError("the controlled protocol expects state_dim=4 and latent_dim=8")
    identity_action = torch.eye(config.state_dim)
    identity_latent = torch.eye(config.latent_dim)
    if not torch.allclose(ACTION_TRANSFORM.T @ ACTION_TRANSFORM, identity_action, atol=1e-6):
        raise ValueError("ACTION_TRANSFORM is not orthogonal")
    if not torch.allclose(LATENT_TRANSFORM.T @ LATENT_TRANSFORM, identity_latent, atol=1e-6):
        raise ValueError("LATENT_TRANSFORM is not orthogonal")
    generator = torch.Generator().manual_seed(12345)
    state = torch.randn(17, config.state_dim, generator=generator)
    action = torch.randn(17, config.state_dim, generator=generator)
    transformed_sum = _transform(state + action, ACTION_TRANSFORM)
    sum_of_transforms = _transform(state, ACTION_TRANSFORM) + _transform(
        action, ACTION_TRANSFORM
    )
    if not torch.allclose(transformed_sum, sum_of_transforms, atol=1e-6):
        raise ValueError("state/action transform convention is inconsistent with s_next=s+a")


def _assert_finite_record(record: dict[str, object]) -> None:
    """Keep malformed or numerically unstable runs out of the aggregate report."""

    scalar_values = [
        record["best_validation_mse"],
        record["test_one_step_mse"],
        record["composition_error"],
        record["stable_rank"],
        *record["rollout_mse"].values(),  # type: ignore[union-attr]
        *record["equivariance"].values(),  # type: ignore[union-attr]
    ]
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise FloatingPointError(f"non-finite evaluation record: {record['mode']}:{record['seed']}")
    for row in record["history"]:  # type: ignore[union-attr]
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite training history: {record['mode']}:{record['seed']}")


@torch.no_grad()
def _rollout_mse(
    model: EquivarianceWorldModel,
    split: dict[str, torch.Tensor],
    horizons: tuple[int, ...],
) -> dict[str, float]:
    states = split["states"]
    actions = split["actions"]
    truth = model.encode(states.reshape(-1, states.size(-1))).reshape(
        states.size(0), states.size(1), -1
    )
    current = truth[:, 0]
    result: dict[str, float] = {}
    for step in range(actions.size(1)):
        current = model.predict(current, actions[:, step])
        horizon = step + 1
        if horizon in horizons:
            result[str(horizon)] = float((current - truth[:, horizon]).square().mean())
    return result


@torch.no_grad()
def _composition_error(
    model: EquivarianceWorldModel, base: torch.Tensor
) -> float:
    generator = torch.Generator().manual_seed(17)
    first = 0.25 * torch.randn(base.size(0), base.size(1), generator=generator)
    second = 0.25 * torch.randn(base.size(0), base.size(1), generator=generator)
    initial = model.encode(base)
    composed = model.predict(model.predict(initial, first), second)
    target = model.encode(base + first + second)
    denominator = target.square().mean().sqrt().clamp_min(1e-8)
    return float((composed - target).square().mean().sqrt() / denominator)


@torch.no_grad()
def _equivariance_metrics(
    model: EquivarianceWorldModel, split: dict[str, torch.Tensor]
) -> dict[str, float]:
    state = split["state"]
    action = split["action"]
    transformed_state = _transform(state, ACTION_TRANSFORM)
    transformed_action = _transform(action, ACTION_TRANSFORM)
    source = model.encode(state)
    transformed_source = model.encode(transformed_state)
    return equivariance_diagnostics(
        model.predict,
        source,
        action,
        transformed_source,
        transformed_action,
        LATENT_TRANSFORM,
    )


def _fit_model(
    mode: str,
    regularization_weight: float,
    seed: int,
    config: EquivarianceBenchmarkConfig,
) -> TrainedBenchmark:
    if mode not in {"none", "latent", "predictor", "both"}:
        raise ValueError(f"unknown mode: {mode}")
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = _make_one_step_split(config.train_samples, config, generator)
    validation = _make_one_step_split(config.validation_samples, config, generator)
    test = _make_one_step_split(config.test_samples, config, generator)
    rollout = _make_rollout_split(config.rollout_trajectories, config, generator)
    model = EquivarianceWorldModel(config.state_dim, config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    best_validation_mse = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        source = model.encode(train["state"])
        target = model.encode(train["target"])
        transformed_source = model.encode(_transform(train["state"], ACTION_TRANSFORM))
        transformed_action = _transform(train["action"], ACTION_TRANSFORM)
        prediction = model.predict(source, train["action"])
        prediction_loss = (prediction - target).square().mean()
        latent_loss = latent_equivariance_loss(
            source, transformed_source, LATENT_TRANSFORM
        )
        predictor_loss = predictor_equivariance_loss(
            model.predict,
            source,
            train["action"],
            transformed_source,
            transformed_action,
            LATENT_TRANSFORM,
        )
        latent_weight = regularization_weight if mode in {"latent", "both"} else 0.0
        predictor_weight = regularization_weight if mode in {"predictor", "both"} else 0.0
        loss = (
            prediction_loss
            + latent_weight * latent_loss
            + predictor_weight * predictor_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_source = model.encode(validation["state"])
            val_target = model.encode(validation["target"])
            val_prediction = model.predict(val_source, validation["action"])
            validation_mse = float((val_prediction - val_target).square().mean())
        history.append(
            {
                "epoch": float(epoch),
                "prediction_loss": float(prediction_loss.detach()),
                "latent_equivariance_loss": float(latent_loss.detach()),
                "predictor_equivariance_loss": float(predictor_loss.detach()),
                "validation_mse": validation_mse,
            }
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state)
    return TrainedBenchmark(
        model=model,
        train=train,
        validation=validation,
        test=test,
        rollout=rollout,
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
        history=history,
    )


def run_one(
    mode: str,
    regularization_weight: float,
    seed: int,
    config: EquivarianceBenchmarkConfig,
) -> dict[str, object]:
    bundle = _fit_model(mode, regularization_weight, seed, config)
    model = bundle.model
    test = bundle.test
    rollout = bundle.rollout
    with torch.no_grad():
        test_source = model.encode(test["state"])
        test_target = model.encode(test["target"])
        test_prediction = model.predict(test_source, test["action"])
        one_step_mse = float((test_prediction - test_target).square().mean())
        equivariance = _equivariance_metrics(model, test)
        composition = _composition_error(model, rollout["states"][:, 0])
    record = {
        "mode": mode,
        "regularization_weight": regularization_weight,
        "seed": seed,
        "device": "cpu",
        "best_epoch": bundle.best_epoch,
        "epochs_run": len(bundle.history),
        "best_validation_mse": bundle.best_validation_mse,
        "test_one_step_mse": one_step_mse,
        "rollout_mse": _rollout_mse(model, rollout, (1, 2, 4, 8)),
        "composition_error": composition,
        "equivariance": equivariance,
        "stable_rank": _stable_rank(test_source),
        "history": bundle.history,
    }
    _assert_finite_record(record)
    return record


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    def mean_std(values: list[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        if len(values) < 2:
            return mean, 0.0
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, math.sqrt(variance)

    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(
            f"{record['mode']}:{record['regularization_weight']}", []
        ).append(record)
    output: dict[str, object] = {}
    for key, rows in grouped.items():
        one_step_mean, one_step_std = mean_std(
            [float(r["test_one_step_mse"]) for r in rows]
        )
        composition_mean, composition_std = mean_std(
            [float(r["composition_error"]) for r in rows]
        )
        latent_mean, latent_std = mean_std(
            [float(r["equivariance"]["latent_equivariance_error"]) for r in rows]  # type: ignore[index]
        )
        predictor_mean, predictor_std = mean_std(
            [float(r["equivariance"]["predictor_equivariance_error"]) for r in rows]  # type: ignore[index]
        )
        rank_mean, rank_std = mean_std([float(r["stable_rank"]) for r in rows])
        rollout_mean: dict[str, float] = {}
        rollout_std: dict[str, float] = {}
        for horizon in rows[0]["rollout_mse"]:  # type: ignore[union-attr]
            values = [float(r["rollout_mse"][horizon]) for r in rows]  # type: ignore[index]
            rollout_mean[horizon], rollout_std[horizon] = mean_std(values)
        output[key] = {
            "n_seeds": len(rows),
            "test_one_step_mse_mean": one_step_mean,
            "test_one_step_mse_std": one_step_std,
            "rollout_mse_mean": rollout_mean,
            "rollout_mse_std": rollout_std,
            "composition_error_mean": composition_mean,
            "composition_error_std": composition_std,
            "latent_equivariance_error_mean": latent_mean,
            "latent_equivariance_error_std": latent_std,
            "predictor_equivariance_error_mean": predictor_mean,
            "predictor_equivariance_error_std": predictor_std,
            "stable_rank_mean": rank_mean,
            "stable_rank_std": rank_std,
        }
    return output


def _write_markdown(report: dict[str, object], output: Path) -> None:
    lines = [
        "# Predictor-equivariance synthetic ablation",
        "",
        "This is a controlled vector-world mechanism benchmark, not a LeWM, PushT, planning, or GPU result.",
        "The ground-truth transition is `s_next = s + a`, and a 90-degree block rotation transforms both states and actions.",
        "",
        "## Protocol",
        "",
        f"- device: `{report['protocol']['device']}`; seeds: `{report['protocol']['seeds']}`; epochs: `{report['protocol']['epochs']}`",
        f"- modes: `{report['protocol']['modes']}`; weights: `{report['protocol']['weights']}`",
        "- The benchmark contains only prediction MSE plus the declared equivariance intervention; no unrelated representation regularizer is included",
        "- `latent` penalizes `E(Ts) - rho(E(s))`; `predictor` penalizes `P(E(Ts), Ta) - rho(P(E(s), a))`; `both` uses both terms",
        "- checkpoints are selected by held-out one-step prediction MSE, not by equivariance error",
        "- rollout uses random action sequences and exact additive ground truth; composition averages two-action products",
        "",
        "## Aggregate results",
        "",
        "| mode | weight | seeds | one-step MSE | rollout h=8 MSE | composition | latent equiv. error | predictor equiv. error | stable rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | "
            f"{values['test_one_step_mse_mean']:.6g} +/- {values['test_one_step_mse_std']:.2g} | "
            f"{values['rollout_mse_mean']['8']:.6g} +/- {values['rollout_mse_std']['8']:.2g} | "
            f"{values['composition_error_mean']:.6g} +/- {values['composition_error_std']:.2g} | "
            f"{values['latent_equivariance_error_mean']:.6g} +/- {values['latent_equivariance_error_std']:.2g} | "
            f"{values['predictor_equivariance_error_mean']:.6g} +/- {values['predictor_equivariance_error_std']:.2g} | "
            f"{values['stable_rank_mean']:.6g} +/- {values['stable_rank_std']:.2g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This artifact tests whether explicit latent and predictor equivariance interventions are distinguishable in a trainable action-conditioned predictor. It does not establish nonlinear LeWM gains, PushT performance, causal controllability, online planning, or GPU efficiency.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 10.0])
    args = parser.parse_args()
    config = EquivarianceBenchmarkConfig(epochs=args.epochs)
    _validate_protocol(config)
    modes = ["none", "latent", "predictor", "both"]
    records: list[dict[str, object]] = []
    start = time.time()
    for mode in modes:
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(
                    f"[equivariance-benchmark] mode={mode} weight={weight} seed={seed}: "
                    "measuring latent-versus-predictor equivariance"
                )
                records.append(run_one(mode, weight, seed, config))
    report = {
        "protocol": {
            **asdict(config),
            "device": "cpu",
            "seeds": args.seeds,
            "modes": modes,
            "weights": args.weights,
            "action_transform": ACTION_TRANSFORM.tolist(),
            "latent_transform": LATENT_TRANSFORM.tolist(),
            "transition": "s_next = s + a",
            "synthetic_vector_world_only": True,
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
