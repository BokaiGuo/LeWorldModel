"""Alignment-strength phase sweep for the nonlinear local-geometry benchmark.

The sweep isolates global and conditional covariance--Jacobian alignment from
predictor equivariance.  Every seed and weight uses the same generated splits,
model initialization, validation-MSE checkpoint rule, rollout generators, and
trajectory-cost planning candidate set.  The output reports both absolute
readouts and paired deltas against the no-alignment baseline.

This is CPU-only synthetic mechanism evidence.  It is designed to test whether
reducing a local commutator has a monotone functional benefit or instead
creates a strength-dependent trade-off; it is not nonlinear LeWM, PushT,
online-control, or causal-control evidence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        _composition_error,
        _diagnostics,
        _fit_model,
        _rollout_mse,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_planning_horizon_synthetic import (
        _trajectory_planning_proxy,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
        _composition_error,
        _diagnostics,
        _fit_model,
        _rollout_mse,
    )
    from run_nonlinear_planning_horizon_synthetic import _trajectory_planning_proxy


METRICS = (
    "test_one_step_mse",
    "id_rollout_h8_mse",
    "ood_rollout_h8_mse",
    "state_shift_composition_error",
    "planning_trajectory_regret",
    "conditional_local_commutator",
)


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


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        signed_mean = float((values * np.asarray(signs)).mean())
        if abs(signed_mean) >= observed - 1e-15:
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
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
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


def _evaluate(
    model: torch.nn.Module,
    test: dict[str, torch.Tensor],
    seed: int,
    config: NonlinearLocalConfig,
) -> dict[str, Any]:
    with torch.no_grad():
        prediction = model(test["state"], test["action"])
        one_step = float((prediction - test["target"]).square().mean())
    rollout_generator = torch.Generator().manual_seed(seed + 50_000)
    ood_generator = torch.Generator().manual_seed(seed + 60_000)
    planning_generator = torch.Generator().manual_seed(seed + 70_000)
    return {
        "test_one_step_mse": one_step,
        "id_rollout_h8_mse": _rollout_mse(
            model,
            config.rollout_trajectories,
            rollout_generator,
            config.rollout_horizon,
            0.0,
            0.5,
        )["8"],
        "ood_rollout_h8_mse": _rollout_mse(
            model,
            config.rollout_trajectories,
            ood_generator,
            config.rollout_horizon,
            config.ood_state_shift,
            0.5,
        )["8"],
        "state_shift_composition_error": _composition_error(
            model,
            config.rollout_trajectories,
            ood_generator,
            config.ood_state_shift,
        ),
        "planning_trajectory_regret": _trajectory_planning_proxy(
            model,
            config.planning_tasks,
            config.planning_candidates,
            config.planning_horizon,
            planning_generator,
            config.ood_state_shift,
        )["trajectory_regret"],
        "conditional_local_commutator": _diagnostics(model, test, config)[
            "conditional_local_commutator"
        ],
    }


def run_one(
    mode: str,
    alignment_weight: float,
    seed: int,
    base_config: NonlinearLocalConfig,
) -> dict[str, Any]:
    if mode not in {"none", "global", "conditional"}:
        raise ValueError(f"unsupported phase-sweep mode: {mode}")
    config = replace(base_config, alignment_weight=float(alignment_weight))
    model, _, test, best_epoch, best_validation, _ = _fit_model(mode, seed, config)
    metrics = _evaluate(model, test, seed, config)
    record = {
        "mode": mode,
        "alignment_weight": 0.0 if mode == "none" else float(alignment_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_validation),
        "metrics": metrics,
    }
    _finite(record)
    return record


def _absolute_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in sorted({str(row["mode"]) for row in records}):
        for weight in sorted({float(row["alignment_weight"]) for row in records if row["mode"] == mode}):
            rows = [
                row
                for row in records
                if row["mode"] == mode and float(row["alignment_weight"]) == weight
            ]
            key = f"{mode}:{weight:g}"
            summary[key] = {
                "mode": mode,
                "alignment_weight": weight,
                "n_seeds": len(rows),
                **{
                    metric: {
                        "mean": float(np.mean([row["metrics"][metric] for row in rows])),
                        "std": float(np.std([row["metrics"][metric] for row in rows], ddof=1))
                        if len(rows) > 1
                        else 0.0,
                    }
                    for metric in METRICS
                },
            }
    return summary


def run_phase(
    alignment_weights: list[float],
    seeds: list[int],
    *,
    epochs: int = 160,
    train_samples: int = 512,
    validation_samples: int = 256,
    test_samples: int = 1024,
    local_anchors: int = 128,
) -> dict[str, Any]:
    base_config = NonlinearLocalConfig(
        epochs=epochs,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        local_anchors=local_anchors,
    )
    weights = [float(weight) for weight in alignment_weights]
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("phase-sweep alignment weights must be positive")
    records: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        baseline = run_one("none", 0.0, int(seed), base_config)
        records.append(baseline)
        print(
            f"[nonlinear-phase] seed={seed} mode=none "
            f"h8={baseline['metrics']['id_rollout_h8_mse']:.6g}"
        )
        for mode in ("global", "conditional"):
            for weight in weights:
                record = run_one(mode, weight, int(seed), base_config)
                records.append(record)
                print(
                    f"[nonlinear-phase] seed={seed} mode={mode} weight={weight:g} "
                    f"h8={record['metrics']['id_rollout_h8_mse']:.6g}"
                )

    baseline_by_seed = {
        int(row["seed"]): row for row in records if row["mode"] == "none"
    }
    inference: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for mode in ("global", "conditional"):
        for weight in weights:
            rows = [
                row
                for row in records
                if row["mode"] == mode and float(row["alignment_weight"]) == weight
            ]
            for metric_index, metric in enumerate(METRICS):
                deltas = np.asarray(
                    [
                        float(row["metrics"][metric])
                        - float(baseline_by_seed[int(row["seed"])] ["metrics"][metric])
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                key = f"{mode}:{weight:g}:{metric}"
                inference[key] = {
                    "mode": mode,
                    "alignment_weight": weight,
                    "metric": metric,
                    **_paired_summary(deltas, seed=20261003 + metric_index),
                }
                raw_pvalues[key] = inference[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        inference[key]["holm_p_global"] = value

    report = {
        "protocol": {
            **asdict(base_config),
            "alignment_weights": weights,
            "seeds": [int(seed) for seed in seeds],
            "modes": ["none", "global", "conditional"],
            "same_generated_protocol": True,
            "validation_checkpoint_selection": True,
            "planning_metric": "trajectory-cost regret on shared OOD candidate sets",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "absolute_summary": _absolute_summary(records),
        "paired_inference": inference,
        "claim_boundary": [
            "the sweep tests strength-dependent functional trade-offs in a nonlinear synthetic proxy",
            "negative deltas favor the alignment intervention",
            "trajectory planning is a shared-candidate proxy, not online task success or causal control",
            "this does not establish nonlinear LeWM, GPU, or PushT performance",
        ],
    }
    expected = len(seeds) * (1 + 2 * len(weights))
    if len(records) != expected:
        raise RuntimeError(f"incomplete phase matrix: expected {expected}, got {len(records)}")
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Nonlinear local-alignment strength phase sweep",
        "",
        "Every seed and weight uses the same generated splits, validation-MSE checkpoint selection, rollout generators, and OOD trajectory-cost planning candidate set. Values are mean +/- standard deviation over seeds.",
        "",
        "## Absolute readouts",
        "",
        "| mode | weight | n | one-step MSE | ID rollout h=8 | OOD rollout h=8 | composition | planning regret | conditional local C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["absolute_summary"].items():
        def fmt(metric: str) -> str:
            row = values[metric]
            return f"{row['mean']:.6g} +/- {row['std']:.2g}"

        lines.append(
            f"| {values['mode']} | {values['alignment_weight']:.6g} | {values['n_seeds']} | "
            f"{fmt('test_one_step_mse')} | {fmt('id_rollout_h8_mse')} | "
            f"{fmt('ood_rollout_h8_mse')} | {fmt('state_shift_composition_error')} | "
            f"{fmt('planning_trajectory_regret')} | {fmt('conditional_local_commutator')} |"
        )
    lines.extend(
        [
            "",
            "## Paired deltas versus none",
            "",
            "Negative delta favors the intervention. Holm correction is global over all mode-weight-metric tests.",
            "",
            "| mode | weight | metric | n | mean delta | 95% bootstrap CI | exact sign-flip p | global Holm p |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for values in report["paired_inference"].values():
        ci = values["bootstrap_ci95"]
        lines.append(
            f"| {values['mode']} | {values['alignment_weight']:.6g} | {values['metric']} | "
            f"{values['n_paired_seeds']} | {values['mean_delta_target_minus_none']:.6g} | "
            f"[{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | "
            f"{values['holm_p_global']:.6g} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--local-anchors", type=int, default=128)
    args = parser.parse_args()
    report = run_phase(
        args.alignment_weights,
        args.seeds,
        epochs=args.epochs,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        local_anchors=args.local_anchors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    (args.output.parent / "README.md").write_text(
        "# Nonlinear local-alignment strength phase sweep\n\n"
        "Formal 10-seed artifact for the alignment-strength trade-off experiment.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
