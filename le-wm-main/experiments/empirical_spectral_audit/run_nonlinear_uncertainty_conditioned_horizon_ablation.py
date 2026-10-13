"""State-conditioned horizon weighting for the nonlinear world-model proxy.

The fixed multi-horizon sweep showed a short-horizon/OOD versus long-horizon/ID
trade-off.  This experiment asks a more specific question: can the trade-off
be reduced by down-weighting long-horizon losses only at states where a
training-only predictive ensemble is uncertain?

The ensemble is fitted from bootstrap resamples of the training sequences and
is frozen before the compared predictors are trained.  The uncertainty score
therefore uses no validation/test targets.  ``permuted_conditioned`` keeps the
same score distribution but destroys its association with the state, which is
the key falsification control.  This remains a nonlinear synthetic proxy, not
end-to-end LeWM or PushT training.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearWorldModel,
        _seed_everything,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_multihorizon_training_ablation import (
        EVAL_HORIZONS,
        SHIFTS,
        TrainingConfig,
        _fit_condition,
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
    from run_nonlinear_local_alignment_synthetic import NonlinearWorldModel, _seed_everything
    from run_nonlinear_multihorizon_training_ablation import (
        EVAL_HORIZONS,
        SHIFTS,
        TrainingConfig,
        _fit_condition,
        _rollout_model,
        _sequence_metrics,
        _sequence_split,
    )
    from run_nonlinear_planner_metric_ablation import _evaluate_metric, _make_task_batch, _predicted_endpoints


PROFILES = ("one_step", "uniform", "short_favoring", "ensemble_conditioned", "permuted_conditioned")
FAMILIES = ("rollout_mse", "planning_regret", "goal_success_rate", "oracle_choice_agreement")
BASE_WEIGHTS = {1: 1.0, 2: 1.0, 4: 1.0}


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


def _fit_probe_ensemble(
    initial_state: dict[str, torch.Tensor],
    train: dict[str, torch.Tensor],
    config: TrainingConfig,
    seed: int,
    members: int,
    epochs: int,
) -> list[NonlinearWorldModel]:
    """Fit frozen one-step probes on bootstrap resamples of training rows."""

    probes: list[NonlinearWorldModel] = []
    count = int(train["initial"].size(0))
    for member in range(int(members)):
        generator = torch.Generator().manual_seed(310_000 + int(seed) * 100 + member)
        indices = torch.randint(count, (count,), generator=generator)
        model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
        model.load_state_dict(copy.deepcopy(initial_state))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train["initial"][indices], train["actions"][indices, 0])
            objective = (prediction - train["targets"][indices, 0]).square().mean()
            objective.backward()
            optimizer.step()
        probes.append(model)
    return probes


@torch.no_grad()
def _disagreement_scores(
    probes: list[NonlinearWorldModel],
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.stack([probe(states, actions) for probe in probes], dim=0)
    return predictions.var(dim=0, unbiased=False).mean(dim=-1)


def _calibrate_scores(scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Map raw disagreement to [0,1] using training-only reference quantiles."""

    q50 = torch.quantile(reference.detach(), 0.50).clamp_min(1e-10)
    q90 = torch.quantile(reference.detach(), 0.90).clamp_min(q50 + 1e-10)
    return ((scores - q50) / (q90 - q50)).clamp(0.0, 1.0)


def _adaptive_horizon_weights(uncertainty: torch.Tensor) -> dict[int, torch.Tensor]:
    """Keep h=1 stable while making high-uncertainty samples short-horizon-heavy."""

    return {
        1: 1.0 + 0.50 * uncertainty,
        2: torch.ones_like(uncertainty),
        4: 1.0 - 0.75 * uncertainty,
    }


def _fit_profile(
    profile: str,
    initial_state: dict[str, torch.Tensor],
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    config: TrainingConfig,
    uncertainty: torch.Tensor,
) -> tuple[NonlinearWorldModel, int, float, dict[str, float]]:
    if profile == "one_step":
        return _fit_condition("one_step", initial_state, train, validation, config)
    if profile == "uniform":
        return _fit_condition("multi_horizon", initial_state, train, validation, config)
    if profile == "short_favoring":
        fixed = {1: 4.0, 2: 2.0, 4: 1.0}
        sample_weights = {h: torch.full_like(uncertainty, float(value)) for h, value in fixed.items()}
    elif profile in {"ensemble_conditioned", "permuted_conditioned"}:
        sample_weights = _adaptive_horizon_weights(uncertainty)
    else:
        raise ValueError(f"unknown profile: {profile}")

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
            h: (predictions[:, h - 1] - train["targets"][:, h - 1]).square().mean(dim=-1)
            for h in config.validation_horizons
        }
        numerator = torch.stack([(sample_weights[h] * per_horizon[h]).mean() for h in config.validation_horizons]).sum()
        denominator = torch.stack([sample_weights[h].mean() for h in config.validation_horizons]).sum().clamp_min(1e-12)
        objective = numerator / denominator
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
def _selective_error(
    model: NonlinearWorldModel,
    split: dict[str, torch.Tensor],
    probes: list[NonlinearWorldModel],
    reference_scores: torch.Tensor,
) -> dict[str, float]:
    scores = _calibrate_scores(_disagreement_scores(probes, split["initial"], split["actions"][:, 0]), reference_scores)
    predictions = _rollout_model(model, split["initial"], split["actions"])
    output: dict[str, float] = {}
    for horizon in (1, 4):
        errors = (predictions[:, horizon - 1] - split["clean_targets"][:, horizon - 1]).square().mean(dim=-1)
        median = torch.quantile(scores, 0.50)
        low = errors[scores <= median]
        high = errors[scores > median]
        output[f"h{horizon}_uncertainty_error_pearson"] = float(np.corrcoef(scores.numpy(), errors.numpy())[0, 1]) if scores.numel() > 1 else 0.0
        output[f"h{horizon}_high_minus_low_error"] = float(high.mean() - low.mean()) if low.numel() and high.numel() else 0.0
    output["mean_uncertainty"] = float(scores.mean())
    return output


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / (2 ** int(values.size)))


def _paired_summary(values: np.ndarray, seed: int, delta_name: str = "profile_minus_one_step") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(10000)])
    return {
        "n_paired_seeds": int(values.size),
        f"mean_delta_{delta_name}": float(values.mean()),
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


def run_report(config: TrainingConfig, seeds: list[int], probe_members: int = 3, probe_epochs: int = 80) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        _seed_everything(int(seed))
        generator = torch.Generator().manual_seed(320_000 + int(seed))
        train = _sequence_split(config.train_sequences, config.sequence_horizon, generator, config.noise_std)
        validation = _sequence_split(config.validation_sequences, config.sequence_horizon, generator, 0.0)
        test = _sequence_split(config.test_sequences, config.sequence_horizon, generator, 0.0)
        initial_model = NonlinearWorldModel(config.state_dim, config.hidden_dim)
        initial_state = copy.deepcopy(initial_model.state_dict())
        probes = _fit_probe_ensemble(initial_state, train, config, int(seed), probe_members, probe_epochs)
        train_raw = _disagreement_scores(probes, train["initial"], train["actions"][:, 0])
        train_uncertainty = _calibrate_scores(train_raw, train_raw)
        permutation_generator = torch.Generator().manual_seed(330_000 + int(seed))
        permuted_uncertainty = train_uncertainty[torch.randperm(train_uncertainty.numel(), generator=permutation_generator)]
        uncertainty_by_profile = {
            "ensemble_conditioned": train_uncertainty,
            "permuted_conditioned": permuted_uncertainty,
        }
        fitted: dict[str, tuple[NonlinearWorldModel, int, float, dict[str, float]]] = {}
        for profile in PROFILES:
            fitted[profile] = _fit_profile(
                profile,
                initial_state,
                train,
                validation,
                config,
                uncertainty_by_profile.get(profile, torch.zeros_like(train_uncertainty)),
            )
        for profile, (model, best_epoch, validation_score, validation_metrics) in fitted.items():
            test_metrics = _sequence_metrics(model, test, config.validation_horizons)
            rollout_splits: dict[float, dict[str, torch.Tensor]] = {0.0: test}
            ood_generator = torch.Generator().manual_seed(325_000 + int(seed) * 10 + 15)
            rollout_splits[1.5] = _sequence_split(config.test_sequences, 8, ood_generator, 0.0, state_shift=1.5)
            for state_shift in SHIFTS:
                if state_shift == 0.0:
                    rollout_metrics = dict(test_metrics)
                    eval_generator = torch.Generator().manual_seed(325_000 + int(seed) * 10 + 8)
                    rollout_split = _sequence_split(config.test_sequences, 8, eval_generator, 0.0)
                    rollout_metrics["8"] = _sequence_metrics(model, rollout_split, (8,))["8"]
                else:
                    rollout_metrics = _sequence_metrics(model, rollout_splits[state_shift], EVAL_HORIZONS)
                for horizon in EVAL_HORIZONS:
                    rollout = rollout_metrics[str(horizon)]
                    records.append({"seed": int(seed), "profile": profile, "family": "rollout_mse", "state_shift": float(state_shift), "horizon": int(horizon), "value": float(rollout), "best_epoch": int(best_epoch), "validation_score": float(validation_score), "validation_metrics": validation_metrics})
            for shift_index, state_shift in enumerate(SHIFTS):
                for horizon in EVAL_HORIZONS:
                    task_generator = torch.Generator().manual_seed(335_000 + int(seed) * 10_000 + shift_index * 1_000 + horizon)
                    batch = _make_task_batch(config.planning_tasks, config.planning_candidates, horizon, task_generator, state_shift)
                    predicted = _predicted_endpoints(model, batch["initial"], batch["actions"])
                    planning = _evaluate_metric(predicted, batch["true_endpoints"], batch["target_endpoint"], "euclidean", torch.eye(config.state_dim))
                    for family in ("planning_regret", "goal_success_rate", "oracle_choice_agreement"):
                        records.append({"seed": int(seed), "profile": profile, "family": family, "state_shift": float(state_shift), "horizon": int(horizon), "value": float(planning[family]), "best_epoch": int(best_epoch), "validation_score": float(validation_score), "validation_metrics": validation_metrics})
            for shift_index, state_shift in enumerate((0.0, 1.5)):
                eval_generator = torch.Generator().manual_seed(345_000 + int(seed) * 10 + shift_index)
                shifted = _sequence_split(config.test_sequences, 4, eval_generator, 0.0, state_shift=state_shift)
                diag = _selective_error(model, shifted, probes, train_raw)
                selective.append({"seed": int(seed), "profile": profile, "state_shift": float(state_shift), **diag})

    def select(profile: str, family: str, shift: float, horizon: int) -> list[dict[str, Any]]:
        return [row for row in records if row["profile"] == profile and row["family"] == family and float(row["state_shift"]) == float(shift) and int(row["horizon"]) == int(horizon)]

    summary: dict[str, Any] = {}
    for profile in PROFILES:
        summary[profile] = {}
        for family in FAMILIES:
            summary[profile][family] = {}
            shifts = SHIFTS
            for shift in shifts:
                summary[profile][family][str(shift)] = {}
                for horizon in EVAL_HORIZONS:
                    rows = select(profile, family, shift, horizon)
                    if len(rows) != len(seeds):
                        raise RuntimeError(f"incomplete matrix {profile}/{family}/{shift}/{horizon}")
                    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
                    summary[profile][family][str(shift)][str(horizon)] = {"n_seeds": int(values.size), "mean": float(values.mean()), "std": float(values.std(ddof=1))}

    paired: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for profile in PROFILES[1:]:
        for family in FAMILIES:
            shifts = SHIFTS
            for shift in shifts:
                for horizon in EVAL_HORIZONS:
                    one = {row["seed"]: row["value"] for row in select("one_step", family, shift, horizon)}
                    target = {row["seed"]: row["value"] for row in select(profile, family, shift, horizon)}
                    deltas = np.asarray([target[seed] - one[seed] for seed in seeds], dtype=np.float64)
                    key = f"{profile}:{family}:shift={shift:g}:h={horizon}"
                    paired[key] = {"profile": profile, "family": family, "state_shift": float(shift), "horizon": int(horizon), **_paired_summary(deltas, seed=20261030 + horizon)}
                    raw_pvalues[key] = paired[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        paired[key]["holm_p_global"] = value

    conditioned_vs_permuted: dict[str, Any] = {}
    conditioned_pvalues: dict[str, float] = {}
    for family in FAMILIES:
        for shift in SHIFTS:
            for horizon in EVAL_HORIZONS:
                conditioned = {row["seed"]: row["value"] for row in select("ensemble_conditioned", family, shift, horizon)}
                permuted = {row["seed"]: row["value"] for row in select("permuted_conditioned", family, shift, horizon)}
                deltas = np.asarray([conditioned[seed] - permuted[seed] for seed in seeds], dtype=np.float64)
                key = f"{family}:shift={shift:g}:h={horizon}"
                conditioned_vs_permuted[key] = {"family": family, "state_shift": float(shift), "horizon": int(horizon), **_paired_summary(deltas, seed=20261031 + horizon, delta_name="conditioned_minus_permuted")}
                conditioned_pvalues[key] = conditioned_vs_permuted[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(conditioned_pvalues).items():
        conditioned_vs_permuted[key]["holm_p_global"] = value

    report = {
        "protocol": {**asdict(config), "seeds": [int(seed) for seed in seeds], "profiles": list(PROFILES), "families": list(FAMILIES), "evaluation_horizons": list(EVAL_HORIZONS), "state_shifts": [float(value) for value in SHIFTS], "probe_members": int(probe_members), "probe_epochs": int(probe_epochs), "same_data_and_initialization_across_profiles": True, "probe_fit_uses_training_rows_only": True, "checkpoint_selection": "shared validation mean over h=1,2,4", "adaptive_rule": "h1=1+0.50u, h2=1, h4=1-0.75u, u is calibrated ensemble disagreement", "planning_metric": "Euclidean predicted endpoint distance on shared candidate tasks", "synthetic_only": True, "elapsed_seconds": time.time() - started},
        "records": records,
        "selective_diagnostics": selective,
        "summary": summary,
        "paired_inference_vs_one_step": paired,
        "paired_inference_conditioned_vs_permuted": conditioned_vs_permuted,
        "claim_boundary": [
            "This is a nonlinear synthetic state-conditioned loss ablation, not end-to-end LeWM or PushT training.",
            "The disagreement probe is fitted only on training rows and is frozen before the compared predictors are fitted.",
            "The permuted control preserves the uncertainty-score histogram while destroying state conditioning.",
            "The conditioned-versus-permuted comparison is the direct test of whether the state association, rather than the marginal weight distribution, matters.",
            "A planning-regret improvement here is evidence for a synthetic training trade-off, not online control success or a causal LeWM latent-dynamics claim.",
        ],
    }
    direct_h2_rollout = conditioned_vs_permuted["rollout_mse:shift=1.5:h=2"]
    direct_h2_planning = conditioned_vs_permuted["planning_regret:shift=1.5:h=2"]
    direct_h4_planning = conditioned_vs_permuted["planning_regret:shift=1.5:h=4"]
    report["interpretation"] = [
        "The ensemble score separates the ID and shifted distributions descriptively, but state conditioning does not produce a robust planning gain after the permutation control.",
        f"For OOD rollout h=2, ensemble-conditioned minus permuted delta is {direct_h2_rollout['mean_delta_conditioned_minus_permuted']:.6g} with Holm-adjusted p={direct_h2_rollout['holm_p_global']:.6g}.",
        f"For OOD planning regret, conditioned minus permuted deltas are {direct_h2_planning['mean_delta_conditioned_minus_permuted']:.6g} at h=2 and {direct_h4_planning['mean_delta_conditioned_minus_permuted']:.6g} at h=4; both Holm-adjusted comparisons are non-significant.",
        "The shared long-horizon improvement over one-step is therefore consistent with the marginal horizon-weight distribution, not uniquely with state-conditioned dynamics information.",
    ]
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = ["# State-conditioned horizon-weighting ablation", "", "All profiles share sequences, initialization, optimizer, checkpoint selection, candidate sets, and evaluation seeds. The adaptive profile uses a frozen training-only bootstrap ensemble; the permuted control preserves its score distribution but breaks state association.", "", "| profile | family | shift | h | n | mean +/- sd |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for profile in PROFILES:
        for family in FAMILIES:
            shifts = SHIFTS
            for shift in shifts:
                for horizon in EVAL_HORIZONS:
                    values = report["summary"][profile][family][str(shift)][str(horizon)]
                    lines.append(f"| {profile} | {family} | {shift:g} | {horizon} | {values['n_seeds']} | {values['mean']:.6g} +/- {values['std']:.6g} |")
    lines.extend(["", "## Paired inference versus one-step", "", "| profile | family | shift | h | n | mean delta | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
    for values in report["paired_inference_vs_one_step"].values():
        ci = values["bootstrap_ci95"]
        lines.append(f"| {values['profile']} | {values['family']} | {values['state_shift']:g} | {values['horizon']} | {values['n_paired_seeds']} | {values['mean_delta_profile_minus_one_step']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |")
    lines.extend(["", "## Direct state-conditioning test", "", "Negative deltas favor the ensemble-conditioned profile for error/regret metrics; the permutation control has the same marginal score distribution.", "", "| family | shift | h | n | mean delta conditioned - permuted | 95% bootstrap CI | sign-flip p | Holm p |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"])
    for values in report["paired_inference_conditioned_vs_permuted"].values():
        ci = values["bootstrap_ci95"]
        lines.append(f"| {values['family']} | {values['state_shift']:g} | {values['horizon']} | {values['n_paired_seeds']} | {values['mean_delta_conditioned_minus_permuted']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | {values['holm_p_global']:.6g} |")
    lines.extend(["", "## Selective error diagnostics", "", "These are descriptive diagnostics: within each split, samples are divided at the median frozen ensemble disagreement.", "", "| profile | shift | h=1 Pearson | h=1 high-low | h=4 Pearson | h=4 high-low | mean u |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in report["selective_diagnostics"]:
        lines.append(f"| {row['profile']} | {row['state_shift']:g} | {row['h1_uncertainty_error_pearson']:.6g} | {row['h1_high_minus_low_error']:.6g} | {row['h4_uncertainty_error_pearson']:.6g} | {row['h4_high_minus_low_error']:.6g} | {row['mean_uncertainty']:.6g} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["interpretation"]]])
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
    parser.add_argument("--probe-members", type=int, default=3)
    parser.add_argument("--probe-epochs", type=int, default=80)
    args = parser.parse_args()
    config = TrainingConfig(train_sequences=args.train_sequences, validation_sequences=args.validation_sequences, test_sequences=args.test_sequences, sequence_horizon=args.sequence_horizon, epochs=args.epochs, learning_rate=args.learning_rate, noise_std=args.noise_std, planning_tasks=args.planning_tasks, planning_candidates=args.planning_candidates)
    print("Running state-conditioned horizon ablation: fixed weights, ensemble disagreement, and permuted control.")
    report = run_report(config, [int(seed) for seed in args.seeds], probe_members=args.probe_members, probe_epochs=args.probe_epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
