"""Evaluate a calibration-only uncertainty gate for action-conditioned selection.

The gate is deliberately separated from adapter fitting.  Five train-root-only
fits provide an ensemble mean and disagreement.  A small, frozen grid of risk
penalties and support thresholds is selected on calibration roots only.  The
outcome roots are never used to choose a gate, action frame, or weight.

This is an intervention/mechanism artifact, not frozen-LeWM online-planning
evidence: physical terminal states are used only for split reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the repository-root package importable when this file is invoked as a
# script (`python experiments/...py`) rather than with `python -m`.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.decision_benchmark.train_joint_rgb_future_task_adapter import (
    SEEDS,
    _features,
    _fit_one,
    _load_rows,
    _root_ranking,
    _device,
)


LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
QUANTILE_GRID = (0.50, 0.75, 1.00)


def _masks(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        split: np.asarray([str(row["split"]) == split for row in records])
        for split in ("train", "calibration", "outcome")
    }


def _split_data(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        key: value[mask] if isinstance(value, np.ndarray)
        else None if value is None
        else [item for item, keep in zip(value, mask) if keep]
        for key, value in data.items()
    }


def _root_indices(roots: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, root in enumerate(roots):
        grouped.setdefault(str(root), []).append(index)
    return {root: np.asarray(indices, dtype=np.int64) for root, indices in sorted(grouped.items())}


def _root_support(
    train_context: np.ndarray,
    all_context: np.ndarray,
    train_roots: list[str],
    all_roots: list[str],
) -> dict[str, float]:
    # Diagonal standardized distance is intentionally conservative and does
    # not inspect task labels.  It is a context-support diagnostic, not a
    # claim of calibrated probability.
    mean = np.asarray(train_context, dtype=np.float64).mean(axis=0)
    scale = np.maximum(np.asarray(train_context, dtype=np.float64).std(axis=0), 1.0e-3)
    values = np.mean(np.abs((np.asarray(all_context, dtype=np.float64) - mean) / scale), axis=1)
    grouped = _root_indices(all_roots)
    return {root: float(np.mean(values[idx])) for root, idx in grouped.items()}


def _root_rank_instability(
    asym_task_stack: np.ndarray,
    source_xy: np.ndarray,
    goals: np.ndarray,
    roots: list[str],
) -> dict[str, float]:
    """Estimate model-selection instability without using task outcomes.

    Each seed selects the candidate with the lowest predicted goal distance.
    The instability is one minus the frequency of the modal selected
    candidate.  A high value means that the learned representation does not
    support a reproducible action choice even before execution.
    """
    seed_terminal = np.asarray(source_xy, dtype=np.float64)[None, :, :] + np.asarray(
        asym_task_stack, dtype=np.float64,
    )
    seed_cost = np.linalg.norm(seed_terminal - np.asarray(goals, dtype=np.float64)[None, :, :], axis=2)
    selected_by_seed: list[dict[str, int]] = []
    groups = _root_indices(roots)
    for seed_cost_row in seed_cost:
        selected_by_seed.append({
            root: int(idx[int(np.argmin(seed_cost_row[idx]))])
            for root, idx in groups.items()
        })
    result: dict[str, float] = {}
    for root in groups:
        selected = [row[root] for row in selected_by_seed]
        counts = np.unique(np.asarray(selected), return_counts=True)[1]
        result[root] = float(1.0 - np.max(counts) / len(selected))
    return result


def _selection_report(
    data: dict[str, Any],
    mask: np.ndarray,
    predicted_terminal: np.ndarray,
    actual_terminal: np.ndarray,
    score: np.ndarray,
    *,
    success_threshold: float,
) -> dict[str, Any]:
    return _root_ranking(
        [root for root, keep in zip(data["roots"], mask) if keep],
        np.asarray(predicted_terminal, dtype=np.float64)[mask],
        np.asarray(actual_terminal, dtype=np.float64)[mask],
        np.asarray(data["goals"], dtype=np.float64)[mask],
        success_threshold=success_threshold,
        predicted_cost_override=np.asarray(score, dtype=np.float64)[mask],
    )


def _bootstrap(values: list[float], seed: int, draws: int = 20000) -> list[float]:
    values_array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values_array), size=(draws, len(values_array)))
    means = values_array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run(
    dataset: Path,
    rgb_report: Path,
    arrays: Path,
    output: Path,
    *,
    epochs: int = 250,
    device: str = "cuda",
    history_windows: int = 4,
    context_family: str = "rgb_projection",
    action_frame: str = "hybrid",
    success_threshold: float = 0.25,
) -> dict[str, Any]:
    records, rgb_rows, array_payload, metadata = _load_rows(dataset, rgb_report, arrays)
    data = _features(
        records, rgb_rows, array_payload,
        history_windows=history_windows,
        context_family=context_family,
        action_frame=action_frame,
    )
    masks = _masks(records)
    train_data = _split_data(data, masks["train"])
    predictions: dict[str, list[dict[str, Any]]] = {"task_only": [], "asym_joint": []}
    for objective in predictions:
        for seed in SEEDS:
            predictions[objective].append(_fit_one(
                train_data, data, seed=seed, epochs=epochs, device=_device(device),
                objective=objective, rgb_weight=1.0,
                temporal_weight=0.5, ranking_weight=0.5, ranking_margin=0.005,
            ))

    task_stack = np.stack([fit["task_prediction"] for fit in predictions["task_only"]], axis=0)
    asym_stack = np.stack([fit["task_prediction"] for fit in predictions["asym_joint"]], axis=0)
    task_mean = task_stack.mean(axis=0)
    asym_mean = asym_stack.mean(axis=0)
    asym_std = asym_stack.std(axis=0, ddof=1)
    task_terminal = np.asarray(data["source_xy"], dtype=np.float64) + task_mean
    asym_terminal = np.asarray(data["source_xy"], dtype=np.float64) + asym_mean
    goals = np.asarray(data["goals"], dtype=np.float64)
    asym_cost = np.linalg.norm(asym_terminal - goals, axis=1)
    task_cost = np.linalg.norm(task_terminal - goals, axis=1)

    train_context = np.asarray(train_data["context"], dtype=np.float64)
    support_by_root = _root_support(
        train_context, np.asarray(data["context"], dtype=np.float64),
        train_data["roots"], data["roots"],
    )
    support_by_row = np.asarray([support_by_root[str(root)] for root in data["roots"]], dtype=np.float64)
    instability_by_root = _root_rank_instability(
        asym_stack, np.asarray(data["source_xy"]), goals, data["roots"],
    )
    instability_by_row = np.asarray(
        [instability_by_root[str(root)] for root in data["roots"]], dtype=np.float64,
    )

    # Use only label-free scales.  The uncertainty scale is fixed from the
    # training-root ensemble; no calibration/outcome labels enter it.
    train_mask = masks["train"]
    uncertainty = np.linalg.norm(asym_std, axis=1)
    uncertainty_scale = float(np.median(uncertainty[train_mask]))
    uncertainty_scale = max(uncertainty_scale, 1.0e-6)
    uncertainty_normalized = uncertainty / uncertainty_scale
    # The root order is stable and each candidate row shares its root support.
    calibration_root_support = np.asarray(
        [support_by_root[root] for root in sorted(set(train_data["roots"] + [
            str(row["instance_id"]) for row in records if str(row["split"]) == "calibration"
        ])) if root in {str(row["instance_id"]) for row in records if str(row["split"]) == "calibration"}],
        dtype=np.float64,
    )
    if calibration_root_support.size == 0:
        raise ValueError("calibration roots are required for gate selection")
    thresholds = tuple(float(np.quantile(calibration_root_support, q)) for q in QUANTILE_GRID) + (float("inf"),)
    calibration_roots = sorted({
        str(row["instance_id"]) for row in records if str(row["split"]) == "calibration"
    })
    calibration_instability = np.asarray(
        [instability_by_root[root] for root in calibration_roots], dtype=np.float64,
    )
    instability_thresholds = tuple(
        float(np.quantile(calibration_instability, q)) for q in QUANTILE_GRID
    ) + (float("inf"),)

    root_groups = _root_indices(data["roots"])
    calibration_mask = masks["calibration"]
    candidates: list[dict[str, Any]] = []
    for penalty in LAMBDA_GRID:
        penalized = asym_cost + float(penalty) * uncertainty_normalized
        for threshold in thresholds:
            for instability_threshold in instability_thresholds:
                gate = (support_by_row > threshold) | (instability_by_row > instability_threshold)
                score = np.where(gate, task_cost, penalized)
                calibration_report = _selection_report(
                    data, calibration_mask, asym_terminal, np.asarray(data["actual_terminal_xy"]), score,
                    success_threshold=success_threshold,
                )
                candidates.append({
                    "uncertainty_penalty": float(penalty),
                    "support_threshold": float(threshold),
                    "rank_instability_threshold": float(instability_threshold),
                    "fallback_root_count": int(sum(
                        gate[idx].any() for root, idx in root_groups.items()
                        if any(data["roots"][i] == root and calibration_mask[i] for i in idx)
                    )),
                    "calibration": calibration_report,
                    "score": score,
                })
    # Calibration-only selection; ties choose the least intervention.
    candidates.sort(key=lambda item: (
        -float(item["calibration"]["paired_regret_gain_mean"]),
        float(item["uncertainty_penalty"]),
        float(item["support_threshold"]),
        float(item["rank_instability_threshold"]),
    ))
    selected = candidates[0]
    selected_score = np.asarray(selected.pop("score"), dtype=np.float64)
    outcome_report = _selection_report(
        data, masks["outcome"], asym_terminal, np.asarray(data["actual_terminal_xy"]), selected_score,
        success_threshold=success_threshold,
    )
    asym_report = _selection_report(
        data, masks["outcome"], asym_terminal, np.asarray(data["actual_terminal_xy"]), asym_cost,
        success_threshold=success_threshold,
    )
    task_report = _selection_report(
        data, masks["outcome"], task_terminal, np.asarray(data["actual_terminal_xy"]), task_cost,
        success_threshold=success_threshold,
    )
    gain_values = np.asarray(outcome_report["paired_regret_gain_values"], dtype=np.float64)
    result = {
        "schema": "pointmaze_uncertainty_gated_selection.v1",
        "dataset": str(dataset.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "arrays": str(arrays.resolve()),
        "protocol": {
            "fit_split": "train roots only",
            "gate_selection_split": "calibration roots only",
            "outcome_used_for_selection": False,
            "ensemble_seeds": list(SEEDS),
            "action_frame": action_frame,
            "context_family": context_family,
            "uncertainty": "std of five asym_joint task predictions, L2 per candidate",
            "uncertainty_scale": "train-root median ensemble disagreement",
            "support": "diagonal standardized mean absolute context distance",
            "rank_instability": "one minus modal candidate frequency across five train-root-only seeds",
            "lambda_grid": list(LAMBDA_GRID),
            "support_quantile_grid": list(QUANTILE_GRID),
            "rank_instability_quantile_grid": list(QUANTILE_GRID),
            "fallback": "task_only score when support exceeds calibration threshold",
            "device": str(predictions["asym_joint"][0]["training"]["device"]),
        },
        "selected_gate": {
            key: value for key, value in selected.items() if key != "calibration"
        },
        "calibration_candidates": [
            {key: value for key, value in item.items() if key != "score"}
            for item in candidates
        ],
        "outcome": {
            "gated": outcome_report,
            "asym_joint_no_gate": asym_report,
            "task_only_fallback": task_report,
            "gated_bootstrap_ci95": _bootstrap(gain_values.tolist(), seed=20261017),
        },
        "support_by_root": support_by_root,
        "rank_instability_by_root": instability_by_root,
        "claim_boundary": "calibration-only uncertainty/support gate for an intervention-supervised PointMaze adapter; not frozen RGB online-planning evidence or universal transfer",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--history-windows", type=int, default=4)
    parser.add_argument("--context-family", choices=("rgb_projection", "latent_model"), default="rgb_projection")
    parser.add_argument("--action-frame", choices=("raw_control", "goal_relative", "hybrid"), default="hybrid")
    parser.add_argument("--success-threshold", type=float, default=0.25)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({
        "selected_gate": result["selected_gate"],
        "outcome": {
            key: value["ranking"]["paired_regret_gain_mean"]
            for key, value in result["outcome"].items()
            if isinstance(value, dict) and "ranking" in value
        },
        "ci95": result["outcome"]["gated_bootstrap_ci95"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
