"""Training-side composition-aware ablation for the nonlinear world proxy.

The existing multi-horizon ablation supervises h=1,2,4 endpoints uniformly.
This experiment adds a targeted composition intervention: h=2 and h=4 losses
on action-switching prefixes receive extra weight, while data, initialization,
optimizer, validation checkpoint selection, and evaluation candidates remain
matched.  The intervention is intended to test whether making action
composition explicit transfers to mixed-action rollout and a planning proxy.

This is a nonlinear synthetic training proxy.  It is not end-to-end LeWM,
PushT, online control, or GPU evidence.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearWorldModel,
        _seed_everything,
        true_transition,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import (
        TrainingConfig,
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
    from run_nonlinear_local_alignment_synthetic import NonlinearWorldModel, _seed_everything, true_transition
    from run_nonlinear_multihorizon_training_ablation import TrainingConfig, _rollout_model, _sequence_metrics, _sequence_split
    from run_nonlinear_planner_metric_ablation import _evaluate_metric, _make_task_batch, _predicted_endpoints


CONDITIONS = ("one_step", "multi_horizon", "composition_aware")
EVAL_HORIZONS = (1, 2, 4, 8)
SHIFTS = (0.0, 1.5)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _switch_mask(actions: torch.Tensor, horizon: int) -> torch.Tensor:
    if horizon <= 1:
        return torch.zeros(actions.size(0), dtype=torch.bool, device=actions.device)
    return (actions[:, 1:horizon] != actions[:, : horizon - 1]).any(dim=1)


def _fit_condition(
    condition: str,
    initial_state: dict[str, torch.Tensor],
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: TrainingConfig,
    composition_weight: float,
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
        per_horizon = {
            horizon: (predictions[:, horizon - 1] - train["targets"][:, horizon - 1]).square().mean(dim=-1)
            for horizon in config.validation_horizons
        }
        if condition == "one_step":
            objective = per_horizon[1].mean()
        else:
            terms = []
            for horizon in config.validation_horizons:
                if condition == "composition_aware" and horizon in (2, 4):
                    mask = _switch_mask(train["actions"], horizon).float()
                    weights = 1.0 + float(composition_weight) * mask
                    terms.append((weights * per_horizon[horizon]).mean() / weights.mean().clamp_min(1e-12))
                else:
                    terms.append(per_horizon[horizon].mean())
            objective = torch.stack(terms).mean()
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


@torch.no_grad()
def _composition_metrics(model: NonlinearWorldModel, split: dict[str, torch.Tensor], horizon: int) -> dict[str, float]:
    predictions = _rollout_model(model, split["initial"], split["actions"])
    errors = (predictions[:, horizon - 1] - split["clean_targets"][:, horizon - 1]).square().mean(dim=-1)
    mask = _switch_mask(split["actions"], horizon)
    if not mask.any() or mask.all():
        raise RuntimeError("composition split must contain both switching and constant prefixes")
    return {
        "switch_mse": float(errors[mask].mean()),
        "constant_mse": float(errors[~mask].mean()),
        "switch_fraction": float(mask.float().mean()),
    }


def _evaluate_condition(
    model: NonlinearWorldModel,
    seed: int,
    config: TrainingConfig,
    test: dict[str, torch.Tensor],
    ood: dict[str, torch.Tensor],
) -> dict[str, float]:
    test_short = _sequence_metrics(model, test, (1, 2, 4))
    id_long = _sequence_split(config.test_sequences, 8, torch.Generator().manual_seed(410_000 + seed), 0.0)
    ood_metrics = _sequence_metrics(model, ood, (1, 2, 4, 8))
    id_long_metrics = _sequence_metrics(model, id_long, (8,))
    composition = _composition_metrics(model, test, 4)
    result = {
        "one_step_mse": float(test_short["1"]),
        "rollout_id_h4_mse": float(test_short["4"]),
        "rollout_id_h8_mse": float(id_long_metrics["8"]),
        "rollout_ood_h2_mse": float(ood_metrics["2"]),
        "rollout_ood_h4_mse": float(ood_metrics["4"]),
        "rollout_ood_h8_mse": float(ood_metrics["8"]),
        "composition_switch_mse": composition["switch_mse"],
        "composition_constant_mse": composition["constant_mse"],
        "composition_switch_minus_constant": composition["switch_mse"] - composition["constant_mse"],
    }
    return result


def run_report(config: TrainingConfig, seeds: list[int], composition_weight: float) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        _seed_everything(int(seed))
        generator = torch.Generator().manual_seed(400_000 + int(seed))
        train = _sequence_split(config.train_sequences, config.sequence_horizon, generator, config.noise_std)
        validation = _sequence_split(config.validation_sequences, config.sequence_horizon, generator, 0.0)
        test = _sequence_split(config.test_sequences, config.sequence_horizon, generator, 0.0)
        ood = _sequence_split(config.test_sequences, 8, torch.Generator().manual_seed(405_000 + int(seed)), 0.0, state_shift=1.5)
        initial = NonlinearWorldModel(config.state_dim, config.hidden_dim)
        initial_state = copy.deepcopy(initial.state_dict())
        for condition in CONDITIONS:
            model, best_epoch, validation_score, validation_metrics = _fit_condition(
                condition, initial_state, train, validation, config, composition_weight
            )
            metrics = _evaluate_condition(model, int(seed), config, test, ood)
            records.append({
                "seed": int(seed),
                "condition": condition,
                "best_epoch": int(best_epoch),
                "validation_score": float(validation_score),
                "validation_metrics": validation_metrics,
                "metrics": metrics,
            })
            print(f"[composition-aware] seed={seed} condition={condition} h8={metrics['rollout_id_h8_mse']:.6g}", flush=True)

    metric_names = list(records[0]["metrics"])
    by_condition: dict[str, dict[int, dict[str, float]]] = {condition: {} for condition in CONDITIONS}
    for row in records:
        by_condition[row["condition"]][int(row["seed"])] = row["metrics"]

    def paired(target: str, baseline: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for index, metric in enumerate(metric_names):
            deltas = [
                by_condition[target][seed][metric] - by_condition[baseline][seed][metric]
                for seed in seeds
            ]
            rng = np.random.default_rng(20261050 + index)
            bootstrap = np.asarray([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(10000)])
            output[metric] = {
                "mean_delta_target_minus_baseline": float(np.mean(deltas)),
                "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                "fraction_target_better": float(np.mean(np.asarray(deltas) < 0.0)),
                "paired_deltas": [float(value) for value in deltas],
            }
        return output

    report = {
        "artifact": "nonlinear_composition_aware_training_ablation",
        "status": "complete",
        "protocol": {
            **{key: value for key, value in config.__dict__.items()},
            "seeds": [int(seed) for seed in seeds],
            "conditions": list(CONDITIONS),
            "composition_weight": float(composition_weight),
            "composition_rule": "extra weight on samples with at least one action switch in the h=2 or h=4 prefix",
            "checkpoint_selection": "shared validation mean over h=1,2,4",
            "planning_proxy": "not added; composition and rollout are the primary intervention outcomes",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "paired_inference_vs_one_step": {
            "multi_horizon": paired("multi_horizon", "one_step"),
            "composition_aware": paired("composition_aware", "one_step"),
            "composition_aware_vs_multi_horizon": paired("composition_aware", "multi_horizon"),
        },
        "claim_boundary": [
            "The intervention changes training weights, not the model architecture or data distribution.",
            "Composition metrics are mixed-action rollout diagnostics, not a physical control or task-success measure.",
            "This does not establish nonlinear LeWM, PushT, online planning, or GPU performance.",
        ],
    }
    if not _finite(report):
        raise FloatingPointError("non-finite composition-aware report")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    rows = report["records"]
    summary: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        summary.setdefault(row["condition"], {})
        for metric, value in row["metrics"].items():
            summary[row["condition"]].setdefault(metric, []).append(float(value))
    lines = [
        "# Nonlinear composition-aware training ablation",
        "",
        "The composition-aware condition adds extra training weight to h=2 and h=4 prefixes containing an action switch. All conditions share seeds, data, initialization, optimizer, and validation checkpoint selection. This is a nonlinear synthetic proxy.",
        "",
        "| condition | seeds | one-step | ID h=4 | ID h=8 | OOD h=4 | OOD h=8 | switch h=4 | constant h=4 | switch-minus-constant |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        values = summary[condition]
        mean = lambda name: sum(values[name]) / len(values[name])
        lines.append(
            f"| {condition} | {len(values['one_step_mse'])} | {mean('one_step_mse'):.6g} | "
            f"{mean('rollout_id_h4_mse'):.6g} | {mean('rollout_id_h8_mse'):.6g} | "
            f"{mean('rollout_ood_h4_mse'):.6g} | {mean('rollout_ood_h8_mse'):.6g} | "
            f"{mean('composition_switch_mse'):.6g} | {mean('composition_constant_mse'):.6g} | "
            f"{mean('composition_switch_minus_constant'):.6g} |"
        )
    lines.extend([
        "",
        "## Paired deltas versus one-step",
        "",
        "Negative values favor the target condition. Bootstrap intervals and per-seed deltas are stored in JSON.",
        "",
        "| condition | metric | mean delta | 95% bootstrap CI | fraction better |",
        "| --- | --- | ---: | --- | ---: |",
    ])
    for condition, metrics in report["paired_inference_vs_one_step"].items():
        for metric, value in metrics.items():
            ci = value["bootstrap_ci95"]
            lines.append(f"| {condition} | {metric} | {value['mean_delta_target_minus_baseline']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['fraction_target_better']:.2f} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--composition-weight", type=float, default=1.0)
    args = parser.parse_args()
    config = TrainingConfig(epochs=args.epochs)
    report = run_report(config, args.seeds, args.composition_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
