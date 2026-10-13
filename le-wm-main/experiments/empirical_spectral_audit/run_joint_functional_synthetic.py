"""Factorial functional test of predictor equivariance and alignment.

This benchmark reuses the joint training path and evaluates all four matched
conditions under the same OOD, rollout, and candidate-set planning protocol:
none, predictor-equivariance only, conditional alignment only, and both.
The interaction is reported separately from the two main effects.  It is
CPU-only synthetic evidence, not nonlinear LeWM, PushT, online control, or
GPU evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

try:
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
    )
    from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import (
        _fit_joint,
    )
    from experiments.empirical_spectral_audit.run_equivariance_functional_synthetic import (
        FunctionalBenchmarkConfig,
        _composition_error,
        _make_one_step_split,
        _make_rollout_split,
        _one_step_mse,
        _planning_proxy,
        _rollout_mse,
    )
    from experiments.empirical_spectral_audit.run_predictor_equivariance_synthetic import (
        EquivarianceBenchmarkConfig,
        _equivariance_metrics,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from experiments.empirical_spectral_audit.regularizers import operator_alignment_diagnostic
    from experiments.empirical_spectral_audit.run_alignment_equivariance_joint_synthetic import _fit_joint
    from experiments.empirical_spectral_audit.run_equivariance_functional_synthetic import (
        FunctionalBenchmarkConfig,
        _composition_error,
        _make_one_step_split,
        _make_rollout_split,
        _one_step_mse,
        _planning_proxy,
        _rollout_mse,
    )
    from experiments.empirical_spectral_audit.run_predictor_equivariance_synthetic import (
        EquivarianceBenchmarkConfig,
        _equivariance_metrics,
    )


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _train_config(config: FunctionalBenchmarkConfig) -> EquivarianceBenchmarkConfig:
    return EquivarianceBenchmarkConfig(
        train_samples=config.train_samples,
        validation_samples=config.validation_samples,
        test_samples=config.test_samples,
        rollout_trajectories=config.rollout_trajectories,
        rollout_horizon=config.rollout_horizon,
        epochs=config.train_epochs,
    )


def run_one(
    mode: str,
    seed: int,
    config: FunctionalBenchmarkConfig,
    predictor_weight: float,
    alignment_weight: float,
) -> dict[str, Any]:
    train_config = _train_config(config)
    model, _train, _validation, test, id_rollout, _history, best_epoch, best_val = _fit_joint(
        mode,
        predictor_weight,
        alignment_weight,
        seed,
        train_config,
    )
    generator = torch.Generator().manual_seed(10_000 + seed)
    state_dim = train_config.state_dim
    splits = {
        "action_2x": _make_one_step_split(
            config.test_samples, state_dim, generator, action_scale=config.ood_action_scale
        ),
        "state_shift": _make_one_step_split(
            config.test_samples, state_dim, generator, state_shift=config.ood_state_shift
        ),
        "combined": _make_one_step_split(
            config.test_samples,
            state_dim,
            generator,
            action_scale=config.ood_action_scale,
            state_shift=config.ood_state_shift,
        ),
    }
    rollout_splits = {
        "action_2x": _make_rollout_split(
            config.rollout_trajectories,
            state_dim,
            config.rollout_horizon,
            generator,
            action_scale=config.ood_action_scale,
        ),
        "state_shift": _make_rollout_split(
            config.rollout_trajectories,
            state_dim,
            config.rollout_horizon,
            generator,
            state_shift=config.ood_state_shift,
        ),
        "combined": _make_rollout_split(
            config.rollout_trajectories,
            state_dim,
            config.rollout_horizon,
            generator,
            action_scale=config.ood_action_scale,
            state_shift=config.ood_state_shift,
        ),
    }
    planning_id = _planning_proxy(model, config, generator, action_scale=0.25, state_shift=0.0)
    planning_ood = _planning_proxy(
        model,
        config,
        generator,
        action_scale=config.ood_action_scale,
        state_shift=config.ood_state_shift,
    )
    with torch.no_grad():
        source = model.encode(test["state"])
        target = model.encode(test["target"])
        prediction = model.predict(source, test["action"])
        alignment = operator_alignment_diagnostic(
            source,
            prediction,
            actions=test["action"],
            mode="conditional",
            num_bins=2,
            min_samples=32,
            ridge=1e-3,
        )
    record: dict[str, Any] = {
        "mode": mode,
        "predictor_weight": float(predictor_weight),
        "alignment_weight": float(alignment_weight),
        "seed": int(seed),
        "device": "cpu",
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_val),
        "test_one_step_mse": _one_step_mse(model, test),
        "id_rollout_mse": _rollout_mse(model, id_rollout, (1, 4, 8)),
        "ood_one_step_mse": {
            key: _one_step_mse(model, split) for key, split in splits.items()
        },
        "ood_rollout_mse": {
            key: _rollout_mse(model, split, (1, 4, 8))
            for key, split in rollout_splits.items()
        },
        "composition_error": {
            "in_distribution": _composition_error(model, test["state"], 0.25),
            "ood_action": _composition_error(model, test["state"], config.ood_action_scale),
        },
        "planning": {"in_distribution": planning_id, "ood_combined": planning_ood},
        "diagnostics": {
            "predictor_equivariance_error": float(
                _equivariance_metrics(model, test)["predictor_equivariance_error"]
            ),
            "operator_alignment": float(alignment["operator_alignment"]),
        },
        "protocol": {
            "transition": "s_next = s + a",
            "evaluation": "matched action-scale, state-shift, combined OOD, h=8 rollout, candidate-set planning",
            "planning_success_threshold": config.planning_success_threshold,
            "synthetic_vector_world_only": True,
        },
    }
    if not _finite(record):
        raise FloatingPointError(f"non-finite joint functional record: {mode}:{seed}")
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "test_one_step_mse": lambda row: row["test_one_step_mse"],
        "id_rollout_h8_mse": lambda row: row["id_rollout_mse"]["8"],
        "ood_action_rollout_h8_mse": lambda row: row["ood_rollout_mse"]["action_2x"]["8"],
        "ood_combined_rollout_h8_mse": lambda row: row["ood_rollout_mse"]["combined"]["8"],
        "ood_planning_regret": lambda row: row["planning"]["ood_combined"]["planning_regret"],
        "ood_planning_success": lambda row: row["planning"]["ood_combined"]["goal_success_rate"],
        "predictor_equivariance_error": lambda row: row["diagnostics"]["predictor_equivariance_error"],
        "operator_alignment": lambda row: row["diagnostics"]["operator_alignment"],
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["mode"]), []).append(row)
    output: dict[str, Any] = {}
    for mode, rows in grouped.items():
        output[mode] = {"n_seeds": len(rows)}
        for name, extractor in paths.items():
            values = [float(extractor(row)) for row in rows]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
            output[mode][name] = {"mean": mean, "std": std}
    return output


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Factorial functional ablation: predictor equivariance × alignment",
        "",
        "This is a CPU-only synthetic vector-world benchmark. All four conditions use the same seed-specific training/evaluation protocol; the planning columns are a candidate-set proxy, not online task success.",
        "",
        "| condition | seeds | one-step MSE | ID rollout h=8 | OOD action rollout h=8 | OOD combined rollout h=8 | OOD regret | OOD success | predictor equivariance | operator alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in report["summary"].items():
        def fmt(name: str) -> str:
            return f"{values[name]['mean']:.6g} +/- {values[name]['std']:.2g}"
        lines.append(
            f"| {mode} | {values['n_seeds']} | {fmt('test_one_step_mse')} | {fmt('id_rollout_h8_mse')} | "
            f"{fmt('ood_action_rollout_h8_mse')} | {fmt('ood_combined_rollout_h8_mse')} | "
            f"{fmt('ood_planning_regret')} | {fmt('ood_planning_success')} | "
            f"{fmt('predictor_equivariance_error')} | {fmt('operator_alignment')} |"
        )
    lines.extend([
        "",
        "The four-way comparison is factorial: `none`, predictor-equivariance only, conditional alignment only, and both. The separate inference artifact reports main effects and the interaction contrast `(both - predictor) - (alignment - none)`.",
        "",
        "Evidence boundary: this is synthetic mechanism and functional-proxy evidence only; it does not establish nonlinear LeWM, PushT, online control, causal controllability, or GPU performance.",
    ])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--predictor-weight", type=float, default=1.0)
    parser.add_argument("--alignment-weight", type=float, default=0.003)
    args = parser.parse_args()
    config = FunctionalBenchmarkConfig(train_epochs=args.epochs)
    conditions = [
        ("none", 0.0, 0.0),
        ("predictor", args.predictor_weight, 0.0),
        ("alignment", 0.0, args.alignment_weight),
        ("both", args.predictor_weight, args.alignment_weight),
    ]
    records: list[dict[str, Any]] = []
    start = time.time()
    for mode, predictor_weight, alignment_weight in conditions:
        for seed in args.seeds:
            print(
                f"[joint-functional] mode={mode} predictor_weight={predictor_weight} "
                f"alignment_weight={alignment_weight} seed={seed}: measuring matched OOD rollout and planning",
                flush=True,
            )
            records.append(run_one(mode, seed, config, predictor_weight, alignment_weight))
    report = {
        "protocol": {
            **asdict(config),
            "seeds": args.seeds,
            "predictor_weight": args.predictor_weight,
            "alignment_weight": args.alignment_weight,
            "conditions": [name for name, _, _ in conditions],
            "factorial_design": True,
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
