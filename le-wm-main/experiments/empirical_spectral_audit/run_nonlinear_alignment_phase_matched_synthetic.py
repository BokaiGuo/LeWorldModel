"""Checkpoint-matched alignment-strength phase sweep.

For every seed, the baseline checkpoint is selected by validation one-step MSE.
For each global/conditional alignment weight, the intervention checkpoint is
selected by nearest validation one-step MSE to that baseline.  Test, rollout,
composition, local-geometry, and trajectory-cost planning metrics are computed
only after checkpoint selection.  This is the matched-prediction companion to
``run_nonlinear_alignment_phase_synthetic.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_checkpoint_matched_geometry_synthetic import (
        _evaluate,
        _select_baseline,
        _select_nearest,
        _train_snapshots,
    )
    from experiments.empirical_spectral_audit.run_nonlinear_local_alignment_synthetic import (
        NonlinearLocalConfig,
    )
except ModuleNotFoundError:  # direct invocation from the experiment directory
    from run_checkpoint_matched_geometry_synthetic import (
        _evaluate,
        _select_baseline,
        _select_nearest,
        _train_snapshots,
    )
    from run_nonlinear_local_alignment_synthetic import NonlinearLocalConfig


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


def _metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    return {
        "test_one_step_mse": float(evaluation["test_one_step_mse"]),
        "id_rollout_h8_mse": float(evaluation["id_rollout_h8_mse"]),
        "ood_rollout_h8_mse": float(evaluation["ood_rollout_h8_mse"]),
        "state_shift_composition_error": float(evaluation["state_shift_composition_error"]),
        "planning_trajectory_regret": float(evaluation["planning"]["trajectory_regret"]),
        "conditional_local_commutator": float(
            evaluation["diagnostics"]["conditional_local_commutator"]
        ),
    }


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
    weights = [float(weight) for weight in alignment_weights]
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("phase-sweep alignment weights must be positive")
    config = NonlinearLocalConfig(
        epochs=epochs,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        local_anchors=local_anchors,
        alignment_weight=weights[0],
    )
    records: list[dict[str, Any]] = []
    started = time.time()
    for seed in seeds:
        baseline_test, baseline_snapshots = _train_snapshots("none", int(seed), config)[1:]
        baseline_checkpoint = _select_baseline(baseline_snapshots)
        baseline_evaluation = _evaluate(
            baseline_checkpoint["state_dict"], baseline_test, int(seed), config
        )
        baseline_metrics = _metrics(baseline_evaluation)
        for mode in ("global", "conditional"):
            for weight in weights:
                intervention_config = NonlinearLocalConfig(
                    **{**vars(config), "alignment_weight": weight}
                )
                intervention_test, intervention_snapshots = _train_snapshots(
                    mode, int(seed), intervention_config
                )[1:]
                intervention_checkpoint = _select_nearest(
                    intervention_snapshots,
                    float(baseline_checkpoint["validation_mse"]),
                )
                intervention_evaluation = _evaluate(
                    intervention_checkpoint["state_dict"],
                    intervention_test,
                    int(seed),
                    intervention_config,
                )
                intervention_metrics = _metrics(intervention_evaluation)
                record = {
                    "mode": mode,
                    "alignment_weight": weight,
                    "seed": int(seed),
                    "baseline_epoch": int(baseline_checkpoint["epoch"]),
                    "intervention_epoch": int(intervention_checkpoint["epoch"]),
                    "baseline_validation_mse": float(baseline_checkpoint["validation_mse"]),
                    "intervention_validation_mse": float(
                        intervention_checkpoint["validation_mse"]
                    ),
                    "relative_validation_gap": abs(
                        float(intervention_checkpoint["validation_mse"])
                        - float(baseline_checkpoint["validation_mse"])
                    ) / max(abs(float(baseline_checkpoint["validation_mse"])), 1e-12),
                    "baseline_metrics": baseline_metrics,
                    "intervention_metrics": intervention_metrics,
                    "deltas": {
                        metric: intervention_metrics[metric] - baseline_metrics[metric]
                        for metric in METRICS
                    },
                }
                _finite(record)
                records.append(record)
                print(
                    f"[matched-phase] seed={seed} mode={mode} weight={weight:g} "
                    f"gap={record['relative_validation_gap']:.3g} "
                    f"h8_delta={record['deltas']['id_rollout_h8_mse']:.6g}"
                )

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
                values = np.asarray(
                    [float(row["deltas"][metric]) for row in rows], dtype=np.float64
                )
                key = f"{mode}:{weight:g}:{metric}"
                inference[key] = {
                    "mode": mode,
                    "alignment_weight": weight,
                    "metric": metric,
                    **_paired_summary(values, seed=20261003 + metric_index),
                }
                raw_pvalues[key] = inference[key]["exact_two_sided_sign_flip_p"]
    for key, value in _holm(raw_pvalues).items():
        inference[key]["holm_p_global"] = value

    report = {
        "protocol": {
            "epochs": epochs,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "local_anchors": local_anchors,
            "alignment_weights": weights,
            "seeds": [int(seed) for seed in seeds],
            "modes": ["global", "conditional"],
            "checkpoint_matching": "baseline validation-MSE optimum; intervention nearest validation-MSE checkpoint",
            "selection_does_not_use": [
                "test error",
                "rollout",
                "composition",
                "planning",
                "geometry diagnostics",
            ],
            "planning_metric": "trajectory-cost regret on shared OOD candidate sets",
            "synthetic_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "paired_inference": inference,
        "matching_summary": {
            f"{mode}:{weight:g}": {
                "n_seeds": len(
                    [
                        row
                        for row in records
                        if row["mode"] == mode and float(row["alignment_weight"]) == weight
                    ]
                ),
                "mean_relative_validation_gap": float(
                    np.mean(
                        [
                            row["relative_validation_gap"]
                            for row in records
                            if row["mode"] == mode
                            and float(row["alignment_weight"]) == weight
                        ]
                    )
                ),
                "max_relative_validation_gap": float(
                    max(
                        row["relative_validation_gap"]
                        for row in records
                        if row["mode"] == mode
                        and float(row["alignment_weight"]) == weight
                    )
                ),
            }
            for mode in ("global", "conditional")
            for weight in weights
        },
        "claim_boundary": [
            "checkpoint matching controls observed validation one-step error, not all representation or optimization differences",
            "negative deltas favor the intervention",
            "trajectory planning is a shared-candidate proxy, not online task success or causal control",
            "this does not establish nonlinear LeWM, GPU, or PushT performance",
        ],
    }
    expected = len(seeds) * 2 * len(weights)
    if len(records) != expected:
        raise RuntimeError(f"incomplete matched phase matrix: expected {expected}, got {len(records)}")
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Checkpoint-matched nonlinear alignment phase sweep",
        "",
        "For every seed, the baseline checkpoint is selected by validation one-step MSE and each intervention checkpoint is selected by nearest validation one-step MSE. Downstream metrics are evaluated only after this selection. Values are paired deltas versus none.",
        "",
        "| mode | weight | metric | n | mean delta | 95% bootstrap CI | exact sign-flip p | global Holm p |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for values in report["paired_inference"].values():
        ci = values["bootstrap_ci95"]
        lines.append(
            f"| {values['mode']} | {values['alignment_weight']:.6g} | {values['metric']} | "
            f"{values['n_paired_seeds']} | {values['mean_delta_target_minus_none']:.6g} | "
            f"[{ci[0]:.6g}, {ci[1]:.6g}] | {values['exact_two_sided_sign_flip_p']:.6g} | "
            f"{values['holm_p_global']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Matching audit",
            "",
            "| mode | weight | seeds | mean relative validation gap | max relative validation gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, values in report["matching_summary"].items():
        mode, weight = key.split(":")
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | "
            f"{values['mean_relative_validation_gap']:.6g} | "
            f"{values['max_relative_validation_gap']:.6g} |"
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
        "# Checkpoint-matched nonlinear alignment phase sweep\n\n"
        "Formal artifact for the matched-prediction alignment-strength sweep.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
