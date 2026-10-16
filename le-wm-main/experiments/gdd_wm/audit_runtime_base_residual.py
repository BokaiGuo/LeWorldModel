"""Fit a fail-closed latent residual correction on frozen latent terminal cost.

This is the deployable counterpart of the capacity audit.  The frozen LeWM
terminal cost remains the base.  The deployable default adds only the latent
component after an action-only nuisance fit has been removed; adding the
action-only component back to an already action-conditioned frozen cost is kept
as an explicit ablation.  A correction is enabled only when its calibration
root-regret CI is strictly below the base cost.  Outcome roots are never used
for alpha or gate selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from experiments.gdd_wm.audit_train_only_teacher_capacity import (
    _bootstrap,
    _feature_vector,
    _load_context_rows,
    _macro_mean_regret,
    _macro_root_regrets,
    _root_metrics,
    _fit_linear_residual_models,
)
from experiments.gdd_wm.build_train_only_readout_teacher_package import (
    _action_summary,
    _feature_key,
    _predict,
)


def _apply_trust_region(
    base_cost: np.ndarray,
    correction: np.ndarray,
    group_roots: np.ndarray,
    correction_multiplier: float,
    base_rank_quantile: float,
) -> np.ndarray:
    """Apply a candidate-local, base-cost trust region to a residual.

    The residual is allowed to change only the base-cost elite set within each
    context/root cell.  This prevents a high-variance learned correction from
    overturning the frozen model on actions outside the region where the base
    model already considers them plausible.  The same operation is mirrored by
    ``CapacityReadoutCostModel`` at runtime.
    """
    if not 0.0 <= correction_multiplier <= 1.0:
        raise ValueError("correction_multiplier must be in [0, 1]")
    if not 0.0 < base_rank_quantile <= 1.0:
        raise ValueError("base_rank_quantile must be in (0, 1]")
    base_cost = np.asarray(base_cost, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    group_roots = np.asarray(group_roots)
    if base_cost.shape != correction.shape or base_cost.shape != group_roots.shape:
        raise ValueError("base_cost, correction, and group_roots must have matching shapes")
    result = base_cost.copy()
    for group in np.unique(group_roots):
        indices = np.flatnonzero(group_roots == group)
        cutoff = float(np.quantile(base_cost[indices], base_rank_quantile))
        keep = base_cost[indices] <= cutoff
        result[indices[keep]] += correction_multiplier * correction[indices[keep]]
    return result


def run(
    *,
    candidate_datasets: list[Path],
    arrays_paths: list[Path],
    output: Path,
    draws: int = 4000,
    seed: int = 20261016,
    target_mode: str = "root_centered",
    goal_conditioning: str = "latent_relative",
    correction_mode: str = "latent_only",
    allow_single_context: bool = False,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    rows, contexts, arrays = _load_context_rows(
        candidate_datasets,
        arrays_paths,
        require_multiple_contexts=not allow_single_context,
    )
    lookup = {item["context_id"]: index for index, item in enumerate(contexts)}
    features: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    targets: list[float] = []
    base_costs: list[float] = []
    group_roots: list[str] = []
    base_roots: list[str] = []
    splits: list[str] = []
    for row in rows:
        context_index = lookup[str(row["candidate_bank_context_id"])]
        key = _feature_key(row)
        key = key.split("::", 1)[-1] if "::" in key else key
        predicted = np.asarray(arrays[context_index][key], dtype=np.float64)
        goal = np.asarray(row["goal_latent_model"], dtype=np.float64)
        if predicted.shape != goal.shape:
            raise ValueError("predicted and goal latent dimensions do not match")
        features.append(_feature_vector(row, predicted, "raw", goal_conditioning))
        action_features.append(_action_summary(row["action_sequence"]))
        targets.append(float(row["real_task_cost"]))
        base_costs.append(float(np.square(predicted - goal).sum()))
        group_roots.append(f"{row['candidate_bank_context_id']}::{row['instance_id']}")
        base_roots.append(str(row["instance_id"]))
        splits.append(str(row["split"]))

    x = np.asarray(features, dtype=np.float64)
    action_x = np.asarray(action_features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    base_cost = np.asarray(base_costs, dtype=np.float64)
    train_mask = np.asarray([split == "train" for split in splits])
    calibration_mask = np.asarray([split == "calibration" for split in splits])
    outcome_mask = np.asarray([split == "outcome" for split in splits])
    target = y - base_cost
    train_target = target[train_mask].copy()
    if target_mode == "root_centered":
        train_roots = np.asarray(group_roots)[train_mask]
        for root in np.unique(train_roots):
            root_mask = train_roots == root
            train_target[root_mask] -= np.mean(train_target[root_mask])
    elif target_mode != "absolute":
        raise ValueError(f"unsupported target mode: {target_mode!r}")
    if correction_mode not in ("latent_only", "full"):
        raise ValueError(f"unsupported correction mode: {correction_mode!r}")

    alpha_grid = (1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)
    base_calibration = _macro_root_regrets(
        base_cost, y, group_roots, base_roots, calibration_mask, 1.0e-3
    )
    base_outcome = _macro_root_regrets(
        base_cost, y, group_roots, base_roots, outcome_mask, 1.0e-3
    )
    candidate_predictions: dict[str, np.ndarray] = {}
    calibration_scores: dict[str, float] = {}
    candidate_specs: list[dict[str, Any]] = []
    context_ids = np.asarray(
        [str(row["candidate_bank_context_id"]) for row in rows], dtype=object
    )
    calibration_contexts = sorted(set(context_ids[calibration_mask].tolist()))
    fits: dict[float, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    correction_grid = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    base_rank_grid = (0.25, 0.5, 0.75, 1.0)
    for alpha in alpha_grid:
        action_fit, residual_fit = _fit_linear_residual_models(
            x, action_x, train_target, train_mask, alpha
        )
        fits[alpha] = (action_fit, residual_fit)
        latent_prediction = _predict(x, residual_fit)
        correction_prediction = (
            _predict(action_x, action_fit) + latent_prediction
            if correction_mode == "full" else latent_prediction
        )
        for correction_multiplier in correction_grid:
            for base_rank_quantile in base_rank_grid:
                candidate_id = (
                    f"alpha={alpha:g};multiplier={correction_multiplier:g};"
                    f"base_rank_quantile={base_rank_quantile:g}"
                )
                prediction = _apply_trust_region(
                    base_cost,
                    correction_prediction,
                    np.asarray(group_roots),
                    correction_multiplier,
                    base_rank_quantile,
                )
                calibration_scores[candidate_id] = _macro_mean_regret(
                    prediction, y, group_roots, base_roots, calibration_mask, 1.0e-3
                )
                candidate_calibration = _macro_root_regrets(
                    prediction, y, group_roots, base_roots, calibration_mask, 1.0e-3
                )
                common = sorted(set(base_calibration) & set(candidate_calibration))
                delta = np.asarray(
                    [candidate_calibration[root] - base_calibration[root] for root in common],
                    dtype=np.float64,
                )
                interval = _bootstrap(delta, np.random.default_rng(seed + 991), draws)
                context_intervals: dict[str, dict[str, Any]] = {}
                for context_index, context_id in enumerate(calibration_contexts):
                    context_mask = calibration_mask & (context_ids == context_id)
                    context_base = _macro_root_regrets(
                        base_cost, y, group_roots, base_roots, context_mask, 1.0e-3
                    )
                    context_candidate = _macro_root_regrets(
                        prediction, y, group_roots, base_roots, context_mask, 1.0e-3
                    )
                    context_common = sorted(
                        set(context_base) & set(context_candidate)
                    )
                    context_delta = np.asarray(
                        [
                            context_candidate[root] - context_base[root]
                            for root in context_common
                        ],
                        dtype=np.float64,
                    )
                    context_interval = _bootstrap(
                        context_delta,
                        np.random.default_rng(seed + 991 + context_index),
                        draws,
                    )
                    context_intervals[context_id] = {
                        "ci95": context_interval["ci95"],
                        "mean": float(np.mean(context_delta)),
                        "root_count": len(context_common),
                    }
                context_gate_pass = bool(
                    context_intervals
                    and all(
                        values["ci95"][1] < 0.0
                        for values in context_intervals.values()
                    )
                )
                candidate_specs.append({
                    "candidate_id": candidate_id,
                    "alpha": alpha,
                    "correction_multiplier": correction_multiplier,
                    "base_rank_quantile": base_rank_quantile,
                    "correction_mode": correction_mode,
                    "calibration_mean_regret": calibration_scores[candidate_id],
                    "calibration_delta_ci95": interval["ci95"],
                    "calibration_pooled_gate_pass": bool(interval["ci95"][1] < 0.0),
                    "calibration_context_intervals": context_intervals,
                    "calibration_context_consensus_gate": context_gate_pass,
                    "calibration_gate_pass": bool(
                        interval["ci95"][1] < 0.0 and context_gate_pass
                    ),
                })
                candidate_predictions[candidate_id] = prediction
    safe_candidates = [item for item in candidate_specs if item["calibration_gate_pass"]]
    if safe_candidates:
        selected_spec = min(
            safe_candidates,
            key=lambda item: (
                item["calibration_mean_regret"],
                item["alpha"],
                -item["correction_multiplier"],
                item["base_rank_quantile"],
            ),
        )
    else:
        selected_spec = min(
            candidate_specs,
            key=lambda item: (
                item["calibration_mean_regret"],
                item["alpha"],
                -item["correction_multiplier"],
                item["base_rank_quantile"],
            ),
        )
    selected_id = str(selected_spec["candidate_id"])
    selected_alpha = float(selected_spec["alpha"])
    selected_multiplier = float(selected_spec["correction_multiplier"])
    selected_quantile = float(selected_spec["base_rank_quantile"])
    selected_prediction = candidate_predictions[selected_id]
    selected_calibration = _macro_root_regrets(
        selected_prediction, y, group_roots, base_roots, calibration_mask, 1.0e-3
    )
    common = sorted(set(base_calibration) & set(selected_calibration))
    calibration_delta = np.asarray(
        [selected_calibration[root] - base_calibration[root] for root in common],
        dtype=np.float64,
    )
    delta_ci = _bootstrap(calibration_delta, np.random.default_rng(seed + 991), draws)
    gate_enabled = bool(selected_spec["calibration_gate_pass"])
    safe_prediction = selected_prediction if gate_enabled else base_cost

    predictions = {
        "base_cost": base_cost,
        "selected_residual": selected_prediction,
        "calibration_safe_gate": safe_prediction,
    }
    metrics: dict[str, Any] = {}
    baseline_cells: dict[str, dict[str, float | None]] | None = None
    outcome_cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, outcome_mask)
        if included
    }
    rng = np.random.default_rng(seed)
    for name, prediction in predictions.items():
        cells = _root_metrics(prediction, y, group_roots, outcome_mask, 1.0e-3)
        roots = {
            root: metrics_item
            for root, metrics_item in zip(
                sorted(set(outcome_cell_to_root.values())), []
            )
        }
        # Reuse the canonical context-to-root macro aggregation through the
        # public helper used by the capacity audit.
        from experiments.gdd_wm.audit_train_only_teacher_capacity import _macro_root_metrics

        root_metrics = _macro_root_metrics(cells, outcome_cell_to_root)
        consistency = np.asarray([
            float(item["pairwise_consistency"]) for item in root_metrics.values()
        ])
        regrets = np.asarray([float(item["regret"]) for item in root_metrics.values()])
        metrics[name] = {
            "outcome_root_count": len(root_metrics),
            "context_root_cell_count": len(cells),
            "pairwise_consistency": _bootstrap(consistency, rng, draws),
            "mean_regret": _bootstrap(regrets, rng, draws),
            "root_metrics": root_metrics,
            "context_root_metrics": cells,
        }
        if name == "base_cost":
            baseline_cells = cells
    assert baseline_cells is not None
    for name, prediction in predictions.items():
        cells = _root_metrics(prediction, y, group_roots, outcome_mask, 1.0e-3)
        from experiments.gdd_wm.audit_train_only_teacher_capacity import _macro_root_metrics

        roots = _macro_root_metrics(cells, outcome_cell_to_root)
        baseline_roots = _macro_root_metrics(baseline_cells, outcome_cell_to_root)
        metrics[name]["vs_base"] = {
            "regret_delta": _bootstrap(
                np.asarray([
                    float(roots[key]["regret"]) - float(baseline_roots[key]["regret"])
                    for key in sorted(roots)
                ]),
                rng,
                draws,
            )
        }

    action_fit, residual_fit = fits[selected_alpha]
    output.mkdir(parents=True, exist_ok=True)
    readout_path = output / "base_residual_readout.npz"
    np.savez_compressed(
        readout_path,
        action_coefficient=np.asarray(action_fit["coefficients"], dtype=np.float32),
        action_mean=np.asarray(action_fit["mean"], dtype=np.float32),
        action_scale=np.asarray(action_fit["scale"], dtype=np.float32),
        residual_coefficient=np.asarray(residual_fit["coefficients"], dtype=np.float32),
        residual_mean=np.asarray(residual_fit["mean"], dtype=np.float32),
        residual_scale=np.asarray(residual_fit["scale"], dtype=np.float32),
        gate_enabled=np.asarray([int(gate_enabled)], dtype=np.int8),
        base_cost_correction=np.asarray([1], dtype=np.int8),
        action_correction_enabled=np.asarray([int(correction_mode == "full")], dtype=np.int8),
        selected_alpha=np.asarray([selected_alpha], dtype=np.float64),
        correction_multiplier=np.asarray([selected_multiplier], dtype=np.float64),
        base_rank_quantile=np.asarray([selected_quantile], dtype=np.float64),
        feature_family=np.asarray(["source_delta_action"]),
        action_frame=np.asarray(["raw"]),
        goal_conditioning=np.asarray([goal_conditioning]),
        target_mode=np.asarray([target_mode]),
    )
    report = {
        "schema": "gdd_wm_runtime_base_residual_audit.v0.2",
        "status": "completed_cuda_diagnostic",
        "protocol": {
            "fit_split": "train",
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "base_cost": "frozen_lewm_terminal_mse",
            "feature_family": "source_delta_action",
            "action_frame": "raw",
            "goal_conditioning": goal_conditioning,
            "target": "real_task_cost_minus_frozen_base_cost",
            "target_mode": target_mode,
            "correction_mode": correction_mode,
            "alpha_grid": list(alpha_grid),
            "device": "cuda",
            "root_bootstrap_draws": draws,
            "gate_rule": "select only calibration-safe trust-region candidates; pooled and every context-specific paired regret CI95 upper bound must be < 0",
            "calibration_context_count": len(calibration_contexts),
            "single_context_allowed": bool(allow_single_context),
            "correction_multiplier_grid": list(correction_grid),
            "base_rank_quantile_grid": list(base_rank_grid),
            "outcome_used_for_selection": False,
        },
        "selection": {
            "selected_alpha": selected_alpha,
            "selected_candidate_id": selected_id,
            "selected_correction_multiplier": selected_multiplier,
            "selected_base_rank_quantile": selected_quantile,
            "selected_correction_mode": correction_mode,
            "calibration_safe_candidate_count": len(safe_candidates),
            "candidate_count": len(candidate_specs),
            "calibration_mean_regret": calibration_scores,
            "calibration_candidate_table": candidate_specs,
            "base_calibration_mean_regret": float(np.mean(list(base_calibration.values()))),
            "selected_calibration_mean_regret": float(np.mean(list(selected_calibration.values()))),
            "selected_minus_base_regret_delta": {
                "mean": float(np.mean(calibration_delta)),
                "ci95": delta_ci["ci95"],
                "root_count": len(common),
            },
            "gate_enabled": gate_enabled,
            "fallback": not gate_enabled,
            "fallback_target": "frozen_base_cost",
        },
        "summary": {
            "record_count": len(rows),
            "train_root_count": len({base_roots[i] for i in np.flatnonzero(train_mask)}),
            "calibration_root_count": len({base_roots[i] for i in np.flatnonzero(calibration_mask)}),
            "outcome_root_count": len({base_roots[i] for i in np.flatnonzero(outcome_mask)}),
            "feature_dimension": int(x.shape[1]),
        },
        "metrics": metrics,
        "runtime_readout_artifact": str(readout_path),
        "claim_boundary": "runtime-safe residual diagnostic; no universal planning claim",
    }
    (output / "base_residual_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", action="append", required=True)
    parser.add_argument("--arrays", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--target-mode", choices=("absolute", "root_centered"), default="root_centered")
    parser.add_argument("--goal-conditioning", choices=("none", "latent_relative"), default="latent_relative")
    parser.add_argument(
        "--correction-mode", choices=("latent_only", "full"), default="latent_only",
        help="deploy the action-residualized latent correction by default; full is an ablation",
    )
    parser.add_argument(
        "--allow-single-context", action="store_true",
        help=(
            "allow one candidate context when it is the exact online proposal "
            "distribution; default multi-context consensus remains fail-closed"
        ),
    )
    args = parser.parse_args()
    run(
        candidate_datasets=[Path(item) for item in args.candidate_dataset],
        arrays_paths=[Path(item) for item in args.arrays],
        output=args.output,
        draws=args.draws,
        seed=args.seed,
        target_mode=args.target_mode,
        goal_conditioning=args.goal_conditioning,
        correction_mode=args.correction_mode,
        allow_single_context=args.allow_single_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
