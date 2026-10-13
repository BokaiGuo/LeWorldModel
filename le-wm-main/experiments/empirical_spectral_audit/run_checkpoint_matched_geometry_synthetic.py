"""Checkpoint-level matched-prediction geometry intervention.

For each seed, the baseline checkpoint is selected by validation one-step MSE.
Each intervention is then matched to that validation error by selecting the
nearest intervention checkpoint, without looking at rollout or planning.  All
downstream metrics are evaluated only after this selection.  This is the
checkpoint-level version of the synthetic nonlinear local-alignment test.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        NonlinearWorldModel,
        _diagnostics,
        _local_alignment_penalty,
        _rollout_mse,
        _sample_split,
        _secant_jacobian,
        _subsample,
        _composition_error,
        _seed_everything,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planning_horizon_synthetic import (
        _trajectory_planning_proxy,
    )
except ModuleNotFoundError:
    from run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        NonlinearWorldModel,
        _diagnostics,
        _local_alignment_penalty,
        _rollout_mse,
        _sample_split,
        _secant_jacobian,
        _subsample,
        _composition_error,
        _seed_everything,
    )
    from run_nonlinear_planning_horizon_synthetic import _trajectory_planning_proxy


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


def _train_snapshots(
    mode: str, seed: int, config: NonlinearLocalConfig
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]]]:
    if mode not in {"none", "global", "conditional"}:
        raise ValueError(f"unsupported checkpoint-matched mode: {mode}")
    _seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = _sample_split(config.train_samples, generator, config.noise_std)
    validation = _sample_split(config.validation_samples, generator, config.noise_std)
    test = _sample_split(config.test_samples, generator, 0.0)
    model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    snapshots: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        anchors = _subsample(train, config.local_anchors)
        prediction = model(train["state"], train["action"])
        prediction_loss = (prediction - train["target"]).square().mean()
        jacobians = _secant_jacobian(model, anchors["state"], anchors["action"], config.local_eps)
        alignment = _local_alignment_penalty(
            anchors["state"], anchors["action"], jacobians, mode, config.local_min_samples,
            reference_state=train["state"], reference_action=train["action"],
        )
        objective = prediction_loss + config.alignment_weight * alignment
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            validation_prediction = model(validation["state"], validation["action"])
            validation_mse = float((validation_prediction - validation["target"]).square().mean())
        snapshots.append(
            {
                "epoch": epoch,
                "validation_mse": validation_mse,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
        )
    return validation, test, snapshots


def _select_baseline(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    return min(snapshots, key=lambda row: (float(row["validation_mse"]), int(row["epoch"])))


def _select_nearest(snapshots: list[dict[str, Any]], target_mse: float) -> dict[str, Any]:
    return min(
        snapshots,
        key=lambda row: (abs(float(row["validation_mse"]) - target_mse), int(row["epoch"])),
    )


def _evaluate(
    state_dict: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    seed: int,
    config: NonlinearLocalConfig,
) -> dict[str, Any]:
    model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
    model.load_state_dict(state_dict)
    with torch.no_grad():
        prediction = model(test["state"], test["action"])
        one_step = float((prediction - test["target"]).square().mean())
    rollout_generator = torch.Generator().manual_seed(seed + 50_000)
    ood_generator = torch.Generator().manual_seed(seed + 60_000)
    planning_generator = torch.Generator().manual_seed(seed + 70_000)
    return {
        "test_one_step_mse": one_step,
        "id_rollout_h8_mse": _rollout_mse(model, config.rollout_trajectories, rollout_generator, config.rollout_horizon, 0.0, 0.5)["8"],
        "ood_rollout_h8_mse": _rollout_mse(model, config.rollout_trajectories, ood_generator, config.rollout_horizon, config.ood_state_shift, 0.5)["8"],
        "state_shift_composition_error": _composition_error(model, config.rollout_trajectories, ood_generator, config.ood_state_shift),
        "diagnostics": _diagnostics(model, test, config),
        "planning": _trajectory_planning_proxy(model, config.planning_tasks, config.planning_candidates, 4, planning_generator, config.ood_state_shift),
    }


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
        "mean_delta_target_minus_none": float(values.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
        "negative_favors_target": True,
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


def run_report(config: NonlinearLocalConfig, seeds: list[int]) -> dict[str, Any]:
    modes = ["global", "conditional"]
    records: list[dict[str, Any]] = []
    metric_names = [
        "test_one_step_mse",
        "id_rollout_h8_mse",
        "ood_rollout_h8_mse",
        "state_shift_composition_error",
        "planning_trajectory_regret",
        "conditional_local_commutator",
    ]
    for seed in seeds:
        _, baseline_test, baseline_snapshots = _train_snapshots("none", seed, config)
        baseline_checkpoint = _select_baseline(baseline_snapshots)
        baseline_eval = _evaluate(baseline_checkpoint["state_dict"], baseline_test, seed, config)
        for mode in modes:
            _, intervention_test, intervention_snapshots = _train_snapshots(mode, seed, config)
            intervention_checkpoint = _select_nearest(
                intervention_snapshots, float(baseline_checkpoint["validation_mse"])
            )
            intervention_eval = _evaluate(intervention_checkpoint["state_dict"], intervention_test, seed, config)
            baseline_metrics = {
                **{key: baseline_eval[key] for key in metric_names if key in baseline_eval},
                "planning_trajectory_regret": baseline_eval["planning"]["trajectory_regret"],
                "conditional_local_commutator": baseline_eval["diagnostics"]["conditional_local_commutator"],
            }
            intervention_metrics = {
                **{key: intervention_eval[key] for key in metric_names if key in intervention_eval},
                "planning_trajectory_regret": intervention_eval["planning"]["trajectory_regret"],
                "conditional_local_commutator": intervention_eval["diagnostics"]["conditional_local_commutator"],
            }
            deltas = {key: float(intervention_metrics[key] - baseline_metrics[key]) for key in metric_names}
            records.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "baseline_epoch": int(baseline_checkpoint["epoch"]),
                    "intervention_epoch": int(intervention_checkpoint["epoch"]),
                    "baseline_validation_mse": float(baseline_checkpoint["validation_mse"]),
                    "intervention_validation_mse": float(intervention_checkpoint["validation_mse"]),
                    "relative_validation_gap": abs(float(intervention_checkpoint["validation_mse"]) - float(baseline_checkpoint["validation_mse"])) / max(abs(float(baseline_checkpoint["validation_mse"])), 1e-12),
                    "baseline_metrics": baseline_metrics,
                    "intervention_metrics": intervention_metrics,
                    "deltas": deltas,
                }
            )
    inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in modes:
        inference[mode] = {}
        for metric in metric_names:
            values = np.asarray([row["deltas"][metric] for row in records if row["mode"] == mode], dtype=np.float64)
            result = _paired_summary(values, seed=20261003 + len(metric))
            inference[mode][metric] = result
            raw_pvalues[f"{mode}::{metric}"] = result["exact_two_sided_sign_flip_p"]
    adjusted = _holm(raw_pvalues)
    for key, value in adjusted.items():
        mode, metric = key.split("::", 1)
        inference[mode][metric]["holm_p_global"] = value
    protocol_config = dict(vars(config)) if hasattr(config, "__dict__") else {}
    report = {
        "protocol": {
            **protocol_config,
            "seeds": seeds,
            "modes": modes,
            "checkpoint_matching": "baseline validation-MSE optimum; intervention nearest validation-MSE checkpoint",
            "selection_does_not_use": ["test error", "rollout", "composition", "planning", "geometry diagnostics"],
            "synthetic_only": True,
        },
        "records": records,
        "inference": inference,
        "claim_boundary": [
            "checkpoint matching controls observed validation one-step error, not all representation or optimization differences",
            "negative deltas favor the intervention",
            "planning is a trajectory-cost candidate-set proxy, not online task success or causal control",
            "this is not evidence of nonlinear LeWM or PushT performance",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Checkpoint-matched geometry intervention",
        "",
        "For each seed, the baseline checkpoint is selected by validation one-step MSE; each intervention selects the nearest checkpoint using only validation one-step MSE. Downstream metrics are evaluated after matching.",
        "",
        "| comparison | metric | n | mean delta | 95% bootstrap CI | exact sign-flip p | global Holm p |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for mode, metrics in report["inference"].items():
        for metric, values in metrics.items():
            ci = values["bootstrap_ci95"]
            lines.append(
                f"| {mode} vs none | {metric} | {values['n_paired_seeds']} | {values['mean_delta_target_minus_none']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |"
            )
    lines.extend([
        "",
        "## Checkpoint matching audit",
        "",
        "| comparison | mean relative validation gap | max relative gap | baseline epochs | intervention epochs |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for mode in sorted({row["mode"] for row in report["records"]}):
        rows = [row for row in report["records"] if row["mode"] == mode]
        lines.append(
            f"| {mode} vs none | {np.mean([row['relative_validation_gap'] for row in rows]):.6g} | {max(row['relative_validation_gap'] for row in rows):.6g} | {[row['baseline_epoch'] for row in rows]} | {[row['intervention_epoch'] for row in rows]} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
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
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    args = parser.parse_args()
    config = NonlinearLocalConfig(
        epochs=args.epochs,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        local_anchors=args.local_anchors,
        alignment_weight=args.alignment_weight,
        predictor_weight=0.1,
    )
    print("Running checkpoint-level matched geometry intervention: select intervention epochs using validation one-step MSE only.")
    report = run_report(config, [int(seed) for seed in args.seeds])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Checkpoint-matched geometry intervention\n\n"
        "The intervention checkpoint is selected by validation one-step MSE only; downstream metrics are evaluated after selection.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
