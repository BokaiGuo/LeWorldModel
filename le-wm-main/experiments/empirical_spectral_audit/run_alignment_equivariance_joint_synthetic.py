"""Joint ablation of predictor equivariance and spectral operator alignment.

This controlled vector-world experiment asks whether the two interventions
measure complementary structure or simply compete for the same predictor
degrees of freedom. All modes share the same data, optimizer, checkpoint
selection, rollout, and composition metrics. It is synthetic mechanism
evidence, not a LeWM, PushT, planning, GPU, or causal-control result.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

try:
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
        predictor_equivariance_loss,
    )
    from experiments.empirical_spectral_audit.run_predictor_equivariance_synthetic import (
        ACTION_TRANSFORM,
        LATENT_TRANSFORM,
        EquivarianceBenchmarkConfig,
        EquivarianceWorldModel,
        _composition_error,
        _equivariance_metrics,
        _make_one_step_split,
        _make_rollout_split,
        _rollout_mse,
        _seed_everything,
        _transform,
        _validate_protocol,
    )
except ModuleNotFoundError:  # direct script invocation
    from regularizers import operator_alignment_diagnostic, operator_alignment_loss, predictor_equivariance_loss
    from run_predictor_equivariance_synthetic import (
        ACTION_TRANSFORM,
        LATENT_TRANSFORM,
        EquivarianceBenchmarkConfig,
        EquivarianceWorldModel,
        _composition_error,
        _equivariance_metrics,
        _make_one_step_split,
        _make_rollout_split,
        _rollout_mse,
        _seed_everything,
        _transform,
        _validate_protocol,
    )


def _fit_joint(
    mode: str,
    predictor_weight: float,
    alignment_weight: float,
    seed: int,
    config: EquivarianceBenchmarkConfig,
) -> tuple[EquivarianceWorldModel, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, float]], int, float]:
    if mode not in {"none", "predictor", "alignment", "both"}:
        raise ValueError(f"unknown mode: {mode}")
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = _make_one_step_split(config.train_samples, config, generator)
    validation = _make_one_step_split(config.validation_samples, config, generator)
    test = _make_one_step_split(config.test_samples, config, generator)
    rollout = _make_rollout_split(config.rollout_trajectories, config, generator)
    model = EquivarianceWorldModel(config.state_dim, config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        source = model.encode(train["state"])
        target = model.encode(train["target"])
        transformed_source = model.encode(_transform(train["state"], ACTION_TRANSFORM))
        transformed_action = _transform(train["action"], ACTION_TRANSFORM)
        prediction = model.predict(source, train["action"])
        prediction_loss = (prediction - target).square().mean()
        predictor_loss = predictor_equivariance_loss(
            model.predict,
            source,
            train["action"],
            transformed_source,
            transformed_action,
            LATENT_TRANSFORM,
        )
        alignment_loss = operator_alignment_loss(
            source,
            prediction,
            actions=train["action"],
            mode="conditional",
            num_bins=2,
            min_samples=32,
            ridge=1e-3,
        )
        objective = prediction_loss
        if mode in {"predictor", "both"}:
            objective = objective + float(predictor_weight) * predictor_loss
        if mode in {"alignment", "both"}:
            objective = objective + float(alignment_weight) * alignment_loss
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            val_source = model.encode(validation["state"])
            val_target = model.encode(validation["target"])
            val_prediction = model.predict(val_source, validation["action"])
            val_mse = float((val_prediction - val_target).square().mean())
        history.append({
            "epoch": float(epoch),
            "prediction_loss": float(prediction_loss.detach()),
            "predictor_equivariance_loss": float(predictor_loss.detach()),
            "operator_alignment_loss": float(alignment_loss.detach()),
            "validation_mse": val_mse,
        })
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("validation never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, train, validation, test, rollout, history, best_epoch, best_val


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def run_one(
    mode: str,
    seed: int,
    config: EquivarianceBenchmarkConfig,
    predictor_weight: float = 0.0,
    alignment_weight: float = 0.0,
) -> dict[str, Any]:
    model, _train, _validation, test, rollout, history, best_epoch, best_val = _fit_joint(
        mode, predictor_weight, alignment_weight, seed, config
    )
    with torch.no_grad():
        source = model.encode(test["state"])
        target = model.encode(test["target"])
        prediction = model.predict(source, test["action"])
        one_step = float((prediction - target).square().mean())
        alignment = operator_alignment_diagnostic(
            source, prediction, actions=test["action"], mode="conditional", num_bins=2, min_samples=32, ridge=1e-3
        )
    record = {
        "mode": mode,
        "predictor_weight": float(predictor_weight),
        "alignment_weight": float(alignment_weight),
        "seed": int(seed),
        "device": "cpu",
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val),
        "test_one_step_mse": one_step,
        "rollout_mse": _rollout_mse(model, rollout, (1, 2, 4, 8)),
        "composition_error": _composition_error(model, rollout["states"][:, 0], config),
        "equivariance": _equivariance_metrics(model, test),
        "operator_alignment": alignment,
        "history": history,
    }
    if not _finite(record):
        raise FloatingPointError(f"non-finite record: {mode}:{predictor_weight}:{alignment_weight}:{seed}")
    return record


def _mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "std": float(tensor.std(unbiased=values.__len__() > 1))}


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = (
            f"{record['mode']}:predictor={record['predictor_weight']}:"
            f"alignment={record['alignment_weight']}"
        )
        grouped.setdefault(key, []).append(record)
    result: dict[str, Any] = {}
    for key, rows in grouped.items():
        result[key] = {
            "n_seeds": len(rows),
            "mode": rows[0]["mode"],
            "predictor_weight": rows[0]["predictor_weight"],
            "alignment_weight": rows[0]["alignment_weight"],
            "test_one_step_mse": _mean_std([float(row["test_one_step_mse"]) for row in rows]),
            "rollout_h8_mse": _mean_std([float(row["rollout_mse"]["8"]) for row in rows]),
            "composition_error": _mean_std([float(row["composition_error"]) for row in rows]),
            "predictor_equivariance_error": _mean_std([float(row["equivariance"]["predictor_equivariance_error"]) for row in rows]),
            "operator_alignment": _mean_std([float(row["operator_alignment"]["operator_alignment"]) for row in rows]),
        }
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Joint spectral-alignment and predictor-equivariance ablation",
        "",
        "This is a CPU-only synthetic vector-world mechanism experiment. It tests whether conditional covariance--predictor alignment and explicit predictor equivariance provide complementary diagnostics or competing objectives.",
        "",
        "| mode | (predictor weight, alignment weight) | seeds | one-step MSE | rollout h=8 | composition | predictor equiv. error | operator alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for values in report["summary"].values():
        lines.append(
            f"| {values['mode']} | ({float(values['predictor_weight']):.6g}, {float(values['alignment_weight']):.6g}) | {values['n_seeds']} | "
            f"{values['test_one_step_mse']['mean']:.6g} +/- {values['test_one_step_mse']['std']:.2g} | "
            f"{values['rollout_h8_mse']['mean']:.6g} +/- {values['rollout_h8_mse']['std']:.2g} | "
            f"{values['composition_error']['mean']:.6g} +/- {values['composition_error']['std']:.2g} | "
            f"{values['predictor_equivariance_error']['mean']:.6g} +/- {values['predictor_equivariance_error']['std']:.2g} | "
            f"{values['operator_alignment']['mean']:.6g} +/- {values['operator_alignment']['std']:.2g} |"
        )
    lines.extend([
        "",
        "Interpretation boundary: the operator-alignment value and predictor-equivariance error are distinct diagnostics. A lower value is not evidence of better prediction unless one-step, rollout, and composition metrics agree. This artifact does not establish nonlinear LeWM gains, PushT performance, online planning, or causal control.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--both-predictor-weights", type=float, nargs="+", default=[1.0])
    parser.add_argument("--both-alignment-weights", type=float, nargs="+", default=[0.01])
    args = parser.parse_args()
    config = EquivarianceBenchmarkConfig(epochs=args.epochs)
    _validate_protocol(config)
    modes = ["none", "predictor", "alignment", "both"]
    records: list[dict[str, Any]] = []
    start = time.time()
    if len(args.both_predictor_weights) != len(args.both_alignment_weights):
        raise ValueError("--both-predictor-weights and --both-alignment-weights must have equal lengths")
    conditions = [("none", 0.0, 0.0)]
    conditions.extend(("predictor", value, 0.0) for value in args.weights)
    conditions.extend(("alignment", 0.0, value) for value in args.weights)
    conditions.extend(
        ("both", predictor, alignment)
        for predictor, alignment in zip(args.both_predictor_weights, args.both_alignment_weights)
    )
    for mode, predictor_weight, alignment_weight in conditions:
        for seed in args.seeds:
            print(
                f"[joint-benchmark] mode={mode} predictor_weight={predictor_weight} "
                f"alignment_weight={alignment_weight} seed={seed}: measuring prediction, equivariance, and operator alignment",
                flush=True,
            )
            records.append(run_one(mode, seed, config, predictor_weight, alignment_weight))
    report = {
        "protocol": {
            "device": "cpu",
            "seeds": args.seeds,
            "weights": args.weights,
            "both_predictor_weights": args.both_predictor_weights,
            "both_alignment_weights": args.both_alignment_weights,
            "modes": modes,
            "transition": "s_next = s + a",
            "alignment": "conditional covariance--predictor commutator, two action-norm bins",
            "checkpoint_selection": "held-out one-step prediction MSE",
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
