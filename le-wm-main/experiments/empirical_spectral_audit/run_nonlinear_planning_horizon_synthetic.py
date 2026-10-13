"""Planning-horizon sensitivity for the nonlinear local-alignment benchmark."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        _fit_model,
    )
except ModuleNotFoundError:
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig, _fit_model


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


def _paired_stats(values: np.ndarray, seed: int = 20261003, repeats: int = 10000) -> dict[str, Any]:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(repeats)],
        dtype=np.float64,
    )
    return {
        "n_paired_seeds": int(values.size),
        "mean_delta_target_minus_none": float(values.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": float(exceed / (2 ** int(values.size))),
        "negative_favors_target": True,
    }


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return adjusted


@torch.no_grad()
def _trajectory_planning_proxy(
    model: Any,
    count: int,
    candidates: int,
    horizon: int,
    generator: torch.Generator,
    state_shift: float,
) -> dict[str, float]:
    """Select candidates by predicted trajectory cost, not only final endpoint."""

    initial = torch.randn(count, 4, generator=generator) + state_shift
    actions = (torch.rand(count, candidates, horizon, generator=generator) < 0.5).long()
    flat_initial = initial[:, None, :].expand(-1, candidates, -1).reshape(-1, 4)
    flat_actions = actions.reshape(-1, horizon)
    predicted_states = [flat_initial]
    true_states = [flat_initial]
    predicted = flat_initial
    truth = flat_initial
    # Import locally to keep the runner directly executable without changing
    # the benchmark's source-of-truth transition.
    try:
        from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import true_transition
    except ModuleNotFoundError:
        from run_nonlinear_local_alignment_synthetic import true_transition

    for step in range(horizon):
        action = flat_actions[:, step]
        predicted = model(predicted, action)
        truth = true_transition(truth, action)
        predicted_states.append(predicted)
        true_states.append(truth)
    predicted_path = torch.stack(predicted_states[1:], dim=1).reshape(count, candidates, horizon, -1)
    true_path = torch.stack(true_states[1:], dim=1).reshape(count, candidates, horizon, -1)
    target_path = true_path[:, 0]
    predicted_cost = (predicted_path - target_path[:, None]).square().mean(dim=(-1, -2))
    true_cost = (true_path - target_path[:, None]).square().mean(dim=(-1, -2))
    chosen = predicted_cost.argmin(dim=1)
    selected_cost = true_cost[torch.arange(count), chosen]
    oracle_cost, oracle_indices = true_cost.min(dim=1)
    return {
        "trajectory_regret": float((selected_cost - oracle_cost).mean()),
        "trajectory_success_rate": float((selected_cost <= 0.05).float().mean()),
        "trajectory_oracle_agreement": float((chosen == oracle_indices).float().mean()),
        "final_selected_distance": float(
            (true_path[torch.arange(count), chosen, -1] - target_path[:, -1]).square().sum(dim=-1).sqrt().mean()
        ),
    }


def run_report(config: NonlinearLocalConfig, seeds: list[int], horizons: list[int]) -> dict[str, Any]:
    conditions = ["none", "global", "conditional", "predictor", "both"]
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for condition_index, mode in enumerate(conditions):
            model, _train, _test, best_epoch, best_validation, _random_scale = _fit_model(mode, seed, config)
            for horizon in horizons:
                generator = torch.Generator().manual_seed(
                    100_000 + int(seed) * 1_000 + int(horizon)
                )
                planning = _trajectory_planning_proxy(
                    model,
                    config.planning_tasks,
                    config.planning_candidates,
                    horizon,
                    generator,
                    config.ood_state_shift,
                )
                records.append(
                    {
                        "mode": mode,
                        "seed": int(seed),
                        "planning_horizon": int(horizon),
                        "best_epoch": int(best_epoch),
                        "best_validation_mse": float(best_validation),
                        "planning": planning,
                    }
                )
    summary: dict[str, Any] = {}
    for mode in conditions:
        summary[mode] = {}
        for horizon in horizons:
            rows = [row for row in records if row["mode"] == mode and row["planning_horizon"] == horizon]
            summary[mode][str(horizon)] = {
                "n_seeds": len(rows),
                "trajectory_regret_mean": sum(row["planning"]["trajectory_regret"] for row in rows) / len(rows),
                "trajectory_success_rate_mean": sum(row["planning"]["trajectory_success_rate"] for row in rows) / len(rows),
                "trajectory_oracle_agreement_mean": sum(row["planning"]["trajectory_oracle_agreement"] for row in rows) / len(rows),
            }
    by_key = {(row["mode"], row["seed"], row["planning_horizon"]): row for row in records}
    paired_inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in conditions[1:]:
        paired_inference[mode] = {}
        for horizon in horizons:
            values = np.asarray(
                [
                    by_key[(mode, seed, horizon)]["planning"]["trajectory_regret"]
                    - by_key[("none", seed, horizon)]["planning"]["trajectory_regret"]
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            key = f"{mode}@{horizon}"
            paired_inference[mode][str(horizon)] = _paired_stats(values, seed=20261003 + horizon)
            raw_pvalues[key] = paired_inference[mode][str(horizon)]["exact_two_sided_sign_flip_p"]
    adjusted = _holm(raw_pvalues)
    for mode in conditions[1:]:
        for horizon in horizons:
            paired_inference[mode][str(horizon)]["holm_p_global"] = adjusted[f"{mode}@{horizon}"]
    report = {
        "protocol": {
            **asdict(config),
            "seeds": seeds,
            "conditions": conditions,
            "horizons": horizons,
            "ood_only_planning": True,
            "synthetic_only": True,
            "candidate_set_is_shared_by_seed_and_horizon": True,
        },
        "records": records,
        "summary": summary,
        "paired_inference": paired_inference,
        "claim_boundary": [
            "planning is a candidate-set proxy, not online task success or causal control",
            "the sweep tests horizon sensitivity under one nonlinear synthetic transition",
            "this does not establish nonlinear LeWM or PushT planning gains",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear planning-horizon sensitivity",
        "",
        "The same nonlinear synthetic training protocol is evaluated at multiple candidate-set planning horizons. Metrics are OOD state-shift candidate-set proxies, not online task success.",
        "",
        "| condition | horizon | n | trajectory regret | trajectory success | trajectory oracle agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, horizons in report["summary"].items():
        for horizon, values in horizons.items():
            lines.append(
                f"| {mode} | {horizon} | {values['n_seeds']} | {values['trajectory_regret_mean']:.6g} | {values['trajectory_success_rate_mean']:.6g} | {values['trajectory_oracle_agreement_mean']:.6g} |"
            )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    lines.extend([
        "",
        "## Paired regret differences versus none",
        "",
        "| condition | horizon | n | mean delta | 95% bootstrap CI | exact sign-flip p | global Holm p |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ])
    for mode, horizons in report["paired_inference"].items():
        for horizon, values in horizons.items():
            ci = values["bootstrap_ci95"]
            lines.append(
                f"| {mode} | {horizon} | {values['n_paired_seeds']} | {values['mean_delta_target_minus_none']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |"
            )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--local-anchors", type=int, default=128)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--predictor-weight", type=float, default=0.1)
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
    print("Running shared-protocol nonlinear planning-horizon sweep at horizons 2, 4, and 8.")
    report = run_report(config, [int(seed) for seed in args.seeds], [int(horizon) for horizon in args.horizons])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear planning-horizon sensitivity\n\n"
        "Shared-protocol OOD candidate-set planning evaluation for the nonlinear local-alignment benchmark.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
