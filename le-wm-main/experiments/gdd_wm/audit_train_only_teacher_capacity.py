"""Audit whether a fixed nonlinear residual readout adds latent information.

This is a capacity diagnostic, not a teacher or planning result.  A linear
action-summary model is fit on train roots, then both a fixed linear residual
readout and one fixed shallow MLP residual readout are fit on the same
train-only source/delta/action features.  The MLP is intentionally not tuned
on calibration or outcome data and requires CUDA for an actual run. The
explicit action-frame and target-mode controls make raw-coordinate and
root-cost confounds auditable rather than silently folding them into the
latent residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Preserve the repository's direct-script entry point as well as
# ``python -m experiments...``. Direct execution otherwise places only the
# ``experiments/gdd_wm`` directory on sys.path.
if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from experiments.gdd_wm.build_train_only_readout_teacher_package import (
    _action_hash,
    _action_summary,
    _feature_key,
    _fit_ridge,
    _load,
    _predict,
    _record_features,
)


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _pair_consistency(rows: list[tuple[float, float]], margin: float) -> float | None:
    correct = total = 0
    for left_index, (left_score, left_cost) in enumerate(rows):
        for right_score, right_cost in rows[left_index + 1:]:
            score_delta = left_score - right_score
            cost_delta = right_cost - left_cost
            if abs(score_delta) <= margin or abs(cost_delta) <= margin:
                continue
            total += 1
            correct += int((score_delta > 0.0) == (cost_delta > 0.0))
    return None if total == 0 else float(correct / total)


def _root_metrics(
    scores: np.ndarray, costs: np.ndarray, roots: list[str], mask: np.ndarray, margin: float,
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        grouped[roots[int(index)]].append(int(index))
    result: dict[str, dict[str, float | None]] = {}
    for root, indices in sorted(grouped.items()):
        root_scores = scores[indices]
        root_costs = costs[indices]
        # The readouts predict terminal *cost*, so lower is better.  Using
        # argmax here would deliberately select the worst candidate and
        # manufacture a negative planning result.
        selected = int(np.argmin(root_scores))
        result[root] = {
            "pairwise_consistency": _pair_consistency(
                [(float(root_scores[i]), float(root_costs[i])) for i in range(len(indices))], margin
            ),
            "regret": float(root_costs[selected] - np.min(root_costs)),
        }
    return result


def _macro_root_metrics(
    cell_metrics: dict[str, dict[str, float | None]],
    cell_to_root: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    """Average context-specific cells within each unique root.

    Contexts are repeated diagnostic views of a root. They must never be
    merged into one candidate set before ranking, and must not be counted as
    independent bootstrap roots.
    """
    grouped: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for cell, metrics in cell_metrics.items():
        grouped[cell_to_root[cell]].append(metrics)
    result: dict[str, dict[str, float | None]] = {}
    for root, values in sorted(grouped.items()):
        consistency = [item["pairwise_consistency"] for item in values]
        consistency = [float(value) for value in consistency if value is not None]
        result[root] = {
            "pairwise_consistency": None if not consistency else float(np.mean(consistency)),
            "regret": float(np.mean([float(item["regret"]) for item in values])),
        }
    return result


def _macro_mean_regret(
    prediction: np.ndarray,
    costs: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    mask: np.ndarray,
    margin: float,
) -> float:
    """Return a root-macro regret for a frozen split.

    Calibration contains repeated context views of the same root.  Selecting
    a hyperparameter on the raw row average would therefore overweight roots
    with more contexts; this helper uses the same context-cell then root-macro
    aggregation as the final audit.
    """
    cell = _root_metrics(prediction, costs, group_roots, mask, margin)
    cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, mask)
        if included
    }
    root = _macro_root_metrics(cell, cell_to_root)
    if not root:
        raise ValueError("calibration split contains no complete root cells")
    return float(np.mean([float(item["regret"]) for item in root.values()]))


def _macro_root_regrets(
    prediction: np.ndarray,
    costs: np.ndarray,
    group_roots: list[str],
    base_roots: list[str],
    mask: np.ndarray,
    margin: float,
) -> dict[str, float]:
    cell = _root_metrics(prediction, costs, group_roots, mask, margin)
    cell_to_root = {
        group: base
        for group, base, included in zip(group_roots, base_roots, mask)
        if included
    }
    root = _macro_root_metrics(cell, cell_to_root)
    return {key: float(value["regret"]) for key, value in root.items()}


def _fit_linear_residual_predictions(
    x: np.ndarray,
    action_x: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed action baseline plus a linear latent residual readout."""
    action_fit, residual_fit = _fit_linear_residual_models(
        x, action_x, train_target, train_mask, alpha
    )
    action_pred = _predict(action_x, action_fit)
    return action_pred, action_pred + _predict(x, residual_fit)


def _fit_linear_residual_models(
    x: np.ndarray,
    action_x: np.ndarray,
    train_target: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
):
    """Return the two frozen ridge fits used by the runtime-safe readout."""
    action_fit = _fit_ridge(action_x[train_mask], train_target, alpha)
    action_pred = _predict(action_x, action_fit)
    residual_train = train_target - action_pred[train_mask]
    residual_fit = _fit_ridge(x[train_mask], residual_train, alpha)
    return action_fit, residual_fit


class _ResidualMLP(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _goalframe_action_summary(row: dict[str, Any]) -> np.ndarray:
    source_state = np.asarray(row.get("source_state_raw"), dtype=np.float64)
    goal = np.asarray(row.get("goal_raw"), dtype=np.float64)
    if source_state.ndim != 1 or source_state.size < 2 or goal.ndim != 1 or goal.size < 2:
        raise ValueError("goal-frame action features require source_state_raw and goal_raw")
    direct = goal[:2] - source_state[:2]
    norm = float(np.linalg.norm(direct))
    if norm <= 1.0e-8:
        raise ValueError("goal-frame action features require a nonzero source-goal displacement")
    direct = direct / norm
    perpendicular = np.asarray([-direct[1], direct[0]], dtype=np.float64)
    actions = np.asarray(row["action_sequence"], dtype=np.float64)
    canonical = np.stack([actions @ direct, actions @ perpendicular], axis=1)
    midpoint = max(1, canonical.shape[0] // 2)
    second_half = canonical[midpoint:] if midpoint < canonical.shape[0] else canonical[-1:]
    return np.concatenate([
        np.mean(canonical, axis=0),
        np.std(canonical, axis=0),
        canonical[0],
        canonical[-1],
        np.mean(canonical[:midpoint], axis=0),
        np.mean(second_half, axis=0),
    ])


def _feature_vector(
    row: dict[str, Any], predicted: np.ndarray, action_frame: str,
    goal_conditioning: str = "none",
) -> np.ndarray:
    if action_frame == "raw":
        base = _record_features(row, predicted, "source_delta_action")
    elif action_frame == "goal_relative":
        source = np.asarray(row.get("source_latent_model"), dtype=np.float64)
        if source.ndim != 1 or source.shape != predicted.shape:
            raise ValueError("source_latent_model and predicted latent dimensions do not match")
        base = np.concatenate([source, predicted - source, _goalframe_action_summary(row)])
    else:
        raise ValueError(f"unsupported action frame: {action_frame!r}")
    if goal_conditioning == "none":
        return base
    if goal_conditioning != "latent_relative":
        raise ValueError(f"unsupported goal conditioning: {goal_conditioning!r}")
    source = np.asarray(row.get("source_latent_model"), dtype=np.float64)
    goal = np.asarray(row.get("goal_latent_model"), dtype=np.float64)
    if source.ndim != 1 or source.shape != predicted.shape or goal.shape != predicted.shape:
        raise ValueError("goal_latent_model must match source/predicted latent dimensions")
    return np.concatenate([base, goal - source, predicted - goal])


def _load_context_rows(
    candidate_datasets: list[Path], arrays_paths: list[Path],
    *, require_multiple_contexts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    minimum = 2 if require_multiple_contexts else 1
    if len(candidate_datasets) < minimum or len(candidate_datasets) != len(arrays_paths):
        if require_multiple_contexts:
            raise ValueError("at least two candidate datasets and matching arrays files are required")
        raise ValueError("at least one candidate dataset and matching arrays file are required")
    loaded = [_load(path) for path in candidate_datasets]
    maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for path, (_, source_rows) in zip(candidate_datasets, loaded):
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for row in source_rows:
            key = (str(row["instance_id"]), _action_hash(row))
            if key in current:
                raise ValueError(f"{path}: duplicate root/action pair {key}")
            current[key] = row
        maps.append(current)
    common = set(maps[0])
    for current in maps[1:]:
        common &= set(current)
    if not common:
        raise ValueError("candidate datasets have no common root/action records")
    arrays = [np.load(path, allow_pickle=False) for path in arrays_paths]
    contexts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (path, (metadata, _)) in enumerate(zip(candidate_datasets, loaded)):
        source_hash = str(metadata.get("candidate_bank_hash", "")) or hashlib.sha256(path.read_bytes()).hexdigest()[:20]
        context_id = f"capacity_context_{index:02d}_{source_hash}"
        contexts.append({"context_id": context_id, "source": str(path.resolve())})
        for root, action_hash in sorted(common):
            row = dict(maps[index][(root, action_hash)])
            row["instance_id"] = root
            row["candidate_id"] = f"action_{action_hash}"
            row["candidate_bank_context_id"] = context_id
            row["teacher_set_id"] = f"{root}__{context_id}"
            row["predicted_latent_ref"] = f"{context_id}::{row['predicted_latent_ref']}"
            rows.append(row)
    return rows, contexts, arrays


def run(
    *, candidate_datasets: list[Path], arrays_paths: list[Path], output: Path,
    epochs: int = 800, learning_rate: float = 1.0e-2, weight_decay: float = 1.0e-3,
    margin: float = 1.0e-3, draws: int = 10000, seed: int = 20261016,
    action_frame: str = "raw", target_mode: str = "absolute",
    goal_conditioning: str = "none",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this capacity audit; refusing CPU fallback")
    rows, contexts, arrays = _load_context_rows(candidate_datasets, arrays_paths)
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    targets: list[float] = []
    group_roots: list[str] = []
    base_roots: list[str] = []
    splits: list[str] = []
    context_lookup = {item["context_id"]: index for index, item in enumerate(contexts)}
    for row in rows:
        context_index = context_lookup[str(row["candidate_bank_context_id"])]
        key = _feature_key(row)
        key = key.split("::", 1)[-1] if "::" in key else key
        value = np.asarray(arrays[context_index][key], dtype=np.float64)
        features.append(_feature_vector(row, value, action_frame, goal_conditioning))
        actions.append(
            _action_summary(row["action_sequence"])
            if action_frame == "raw"
            else _goalframe_action_summary(row)
        )
        targets.append(float(row["real_task_cost"]))
        context_id = str(row["candidate_bank_context_id"])
        group_roots.append(f"{context_id}::{row['instance_id']}")
        base_roots.append(str(row["instance_id"]))
        splits.append(str(row["split"]))
    x = np.asarray(features, dtype=np.float64)
    action_x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    train_mask = np.asarray([split == "train" for split in splits])
    calibration_mask = np.asarray([split == "calibration" for split in splits])
    outcome_mask = np.asarray([split == "outcome" for split in splits])
    if target_mode == "absolute":
        train_target = y[train_mask]
    elif target_mode == "root_centered":
        train_target = y[train_mask].copy()
        train_roots = np.asarray(group_roots)[train_mask]
        for root in np.unique(train_roots):
            root_mask = train_roots == root
            train_target[root_mask] -= np.mean(train_target[root_mask])
    else:
        raise ValueError(f"unsupported target mode: {target_mode!r}")
    action_pred, linear_pred = _fit_linear_residual_predictions(
        x, action_x, train_target, train_mask, 1.0e-2
    )
    residual_train = train_target - action_pred[train_mask]

    feature_mean = np.mean(x[train_mask], axis=0)
    feature_scale = np.maximum(np.std(x[train_mask], axis=0), 1.0e-6)
    residual_mean = float(np.mean(residual_train))
    residual_scale = max(float(np.std(residual_train)), 1.0e-6)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    model = _ResidualMLP(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tx = torch.as_tensor((x[train_mask] - feature_mean) / feature_scale, dtype=torch.float32, device=device)
    ty = torch.as_tensor((residual_train - residual_mean) / residual_scale, dtype=torch.float32, device=device)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(tx) - ty) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        all_tx = torch.as_tensor((x - feature_mean) / feature_scale, dtype=torch.float32, device=device)
        residual_pred_mlp = model(all_tx).cpu().numpy() * residual_scale + residual_mean
    mlp_pred = action_pred + residual_pred_mlp

    predictions = {"action_only": action_pred, "linear_residual": linear_pred, "mlp_residual": mlp_pred}

    calibration_selection: dict[str, Any] | None = None
    selected_alpha_for_artifact = 1.0e-2
    gate_passed = False
    if calibration_mask.any():
        # The grid is fixed before looking at outcome roots.  Calibration is
        # used only to control the high-dimensional latent residual's
        # variance; outcome remains untouched until the final report.
        alpha_grid = (1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)
        calibration_scores: dict[str, float] = {}
        candidate_predictions: dict[float, np.ndarray] = {}
        action_calibration_roots = _macro_root_regrets(
            action_pred,
            y,
            group_roots,
            base_roots,
            calibration_mask,
            margin,
        )
        for candidate_alpha in alpha_grid:
            _, candidate_prediction = _fit_linear_residual_predictions(
                x, action_x, train_target, train_mask, candidate_alpha
            )
            candidate_predictions[candidate_alpha] = candidate_prediction
            calibration_scores[str(candidate_alpha)] = _macro_mean_regret(
                candidate_prediction,
                y,
                group_roots,
                base_roots,
                calibration_mask,
                margin,
            )
        selected_alpha = min(
            alpha_grid,
            key=lambda candidate: (calibration_scores[str(candidate)], candidate),
        )
        selected_alpha_for_artifact = float(selected_alpha)
        predictions["calibrated_linear_residual"] = candidate_predictions[selected_alpha]
        selected_calibration_roots = _macro_root_regrets(
            candidate_predictions[selected_alpha],
            y,
            group_roots,
            base_roots,
            calibration_mask,
            margin,
        )
        common_calibration_roots = sorted(
            set(action_calibration_roots) & set(selected_calibration_roots)
        )
        calibration_delta = np.asarray(
            [
                selected_calibration_roots[root] - action_calibration_roots[root]
                for root in common_calibration_roots
            ],
            dtype=np.float64,
        )
        calibration_delta_ci = _bootstrap(
            calibration_delta,
            np.random.default_rng(seed + 991),
            draws,
        )
        gate_passed = bool(calibration_delta_ci["ci95"][1] < 0.0)
        predictions["calibration_safe_gate"] = (
            candidate_predictions[selected_alpha] if gate_passed else action_pred
        )
        calibration_selection = {
            "alpha_grid": [float(value) for value in alpha_grid],
            "calibration_mean_regret": calibration_scores,
            "selected_alpha": float(selected_alpha),
            "action_only_mean_regret": float(np.mean(list(action_calibration_roots.values()))),
            "selected_mean_regret": float(np.mean(list(selected_calibration_roots.values()))),
            "selected_minus_action_regret_delta": {
                "mean": float(np.mean(calibration_delta)),
                "ci95": calibration_delta_ci["ci95"],
                "root_count": len(common_calibration_roots),
            },
            "safe_gate": {
                "rule": "enable residual only when calibration paired regret CI95 upper bound < 0",
                "passed": gate_passed,
                "fallback": not gate_passed,
                "fallback_target": "action_only",
            },
            "selection_split": "calibration",
            "evaluation_split": "outcome",
            "outcome_used_for_selection": False,
        }
    metrics: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    baseline_cells: dict[str, dict[str, float | None]] | None = None
    outcome_cell_to_root = {
        group: base
        for group, base, is_outcome in zip(
            group_roots, base_roots, outcome_mask
        )
        if is_outcome
    }
    for name, prediction in predictions.items():
        cell_root = _root_metrics(prediction, y, group_roots, outcome_mask, margin)
        root = _macro_root_metrics(cell_root, outcome_cell_to_root)
        consistency = np.asarray([float(item["pairwise_consistency"]) for item in root.values()])
        regrets = np.asarray([float(item["regret"]) for item in root.values()])
        metrics[name] = {
            "outcome_root_count": len(root),
            "context_root_cell_count": len(cell_root),
            "pairwise_consistency": _bootstrap(consistency, rng, draws),
            "mean_regret": _bootstrap(regrets, rng, draws),
            "root_metrics": root,
            "context_root_metrics": cell_root,
        }
        if name == "action_only":
            baseline_cells = cell_root
    assert baseline_cells is not None
    for name, prediction in predictions.items():
        cell_root = _root_metrics(prediction, y, group_roots, outcome_mask, margin)
        root = _macro_root_metrics(cell_root, outcome_cell_to_root)
        baseline_root = _macro_root_metrics(baseline_cells, outcome_cell_to_root)
        consistency_delta = np.asarray([
            float(root[key]["pairwise_consistency"]) - float(baseline_root[key]["pairwise_consistency"])
            for key in sorted(root)
        ])
        regret_delta = np.asarray([
            float(root[key]["regret"]) - float(baseline_root[key]["regret"])
            for key in sorted(root)
        ])
        metrics[name]["vs_action_only"] = {
            "pairwise_consistency_delta": _bootstrap(consistency_delta, rng, draws),
            "regret_delta": _bootstrap(regret_delta, rng, draws),
        }
    output.mkdir(parents=True, exist_ok=True)
    readout_artifact = None
    if action_frame == "raw":
        # This export is deployable from RGB/model tensors alone.  The
        # goal-relative branch is intentionally not exported because its
        # action coordinates require evaluator-only source/goal state.
        action_fit, residual_fit = _fit_linear_residual_models(
            x,
            action_x,
            train_target,
            train_mask,
            selected_alpha_for_artifact,
        )
        readout_path = output / "capacity_readout.npz"
        np.savez_compressed(
            readout_path,
            action_coefficient=np.asarray(action_fit["coefficients"], dtype=np.float32),
            action_mean=np.asarray(action_fit["mean"], dtype=np.float32),
            action_scale=np.asarray(action_fit["scale"], dtype=np.float32),
            residual_coefficient=np.asarray(residual_fit["coefficients"], dtype=np.float32),
            residual_mean=np.asarray(residual_fit["mean"], dtype=np.float32),
            residual_scale=np.asarray(residual_fit["scale"], dtype=np.float32),
            gate_enabled=np.asarray([int(gate_passed)], dtype=np.int8),
            selected_alpha=np.asarray([selected_alpha_for_artifact], dtype=np.float64),
            feature_family=np.asarray(["source_delta_action"]),
            action_frame=np.asarray([action_frame]),
            goal_conditioning=np.asarray([goal_conditioning]),
            target_mode=np.asarray([target_mode]),
        )
        readout_artifact = str(readout_path)
    report = {
        "schema": "gdd_wm_train_only_teacher_capacity_audit.v0.1",
        "status": "completed_cuda_diagnostic",
        "protocol": {
            "fit_split": "train",
            "evaluation_split": "outcome",
            "feature_family": "source_delta_action",
            "action_frame": action_frame,
            "goal_conditioning": goal_conditioning,
            "target_mode": target_mode,
            "action_baseline": "standardized_ridge_train_only",
            "linear_residual": "standardized_ridge_train_only",
            "mlp_residual": {"hidden": [64, 32], "activation": "SiLU", "epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay},
            "family_or_capacity_selection": "not_performed",
            "device": str(device),
            "seed": seed,
            "root_bootstrap_draws": draws,
            "pair_margin": margin,
        },
        "summary": {
            "record_count": len(rows),
            "train_root_count": len({base_roots[i] for i in np.flatnonzero(train_mask)}),
            "outcome_root_count": len({base_roots[i] for i in np.flatnonzero(outcome_mask)}),
            "train_context_root_cell_count": len({group_roots[i] for i in np.flatnonzero(train_mask)}),
            "outcome_context_root_cell_count": len({group_roots[i] for i in np.flatnonzero(outcome_mask)}),
            "context_count": len(contexts),
            "feature_dimension": x.shape[1],
        },
        "calibration_selection": calibration_selection,
        "runtime_readout_artifact": readout_artifact,
        "metrics": metrics,
        "claim_boundary": "capacity diagnostic only; no D-JEPA teacher, D2, or planning claim",
    }
    (output / "capacity_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", action="append", required=True)
    parser.add_argument("--arrays", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--action-frame", choices=("raw", "goal_relative"), default="raw")
    parser.add_argument("--target-mode", choices=("absolute", "root_centered"), default="absolute")
    parser.add_argument("--goal-conditioning", choices=("none", "latent_relative"), default="none")
    args = parser.parse_args()
    run(
        candidate_datasets=[Path(item) for item in args.candidate_dataset],
        arrays_paths=[Path(item) for item in args.arrays],
        output=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        draws=args.draws,
        seed=args.seed,
        action_frame=args.action_frame,
        target_mode=args.target_mode,
        goal_conditioning=args.goal_conditioning,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
