"""Compare raw versus goal-relative task-effect charts on one action bank.

The existing hybrid action frame exposes both physical and goal-relative
controls, but the task target is still trained in the raw XY chart.  This
intervention tests the missing coordinate-consistency hypothesis: train the
effect head in the same per-root goal-relative chart, then map predictions
back to physical XY before ranking candidates.  Only source/goal metadata and
candidate actions are used at runtime; terminal outcomes remain evaluation
labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.decision_benchmark.train_joint_rgb_future_task_adapter import (  # noqa: E402
    SEEDS,
    _device,
    _features,
    _fit_one,
    _load_rows,
    _root_ranking,
)


# `asym_rank` computes ranking loss in the adapter's raw XY chart.  Until that
# loss is made frame-aware, including it here would silently mix coordinates;
# keep this intervention limited to objectives whose supervised task loss is
# chart-agnostic.
OBJECTIVES = ("task_only", "asym_joint")


def _split_data(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        key: value[mask] if isinstance(value, np.ndarray)
        else None if value is None
        else [item for item, keep in zip(value, mask) if keep]
        for key, value in data.items()
    }


def _goal_basis(source_xy: np.ndarray, goals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.asarray(goals, dtype=np.float64) - np.asarray(source_xy, dtype=np.float64)
    norm = np.linalg.norm(displacement, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError("goal-relative task frame requires nonzero source-goal displacement")
    direct = displacement / norm
    perpendicular = np.stack([-direct[:, 1], direct[:, 0]], axis=1)
    return direct, perpendicular


def _to_task_frame(values: np.ndarray, source_xy: np.ndarray, goals: np.ndarray, frame: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if frame == "raw_xy":
        return values.copy()
    if frame != "goal_relative":
        raise ValueError(f"unsupported task frame: {frame}")
    direct, perpendicular = _goal_basis(source_xy, goals)
    return np.stack([
        np.sum(values * direct, axis=1),
        np.sum(values * perpendicular, axis=1),
    ], axis=1)


def _from_task_frame(values: np.ndarray, source_xy: np.ndarray, goals: np.ndarray, frame: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if frame == "raw_xy":
        return values.copy()
    direct, perpendicular = _goal_basis(source_xy, goals)
    return values[:, :1] * direct + values[:, 1:2] * perpendicular


def _report(
    data: dict[str, Any],
    mask: np.ndarray,
    predicted_effect: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    return _root_ranking(
        [data["roots"][index] for index in indices],
        np.asarray(data["source_xy"], dtype=np.float64)[indices] + predicted_effect[indices],
        np.asarray(data["actual_terminal_xy"], dtype=np.float64)[indices],
        np.asarray(data["goals"], dtype=np.float64)[indices],
        success_threshold=0.25,
    )


def _root_bootstrap(values: list[float], seed: int = 20261018, draws: int = 20000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
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
    action_frame: str = "hybrid",
) -> dict[str, Any]:
    records, rgb_rows, array_payload, metadata = _load_rows(dataset, rgb_report, arrays)
    data = _features(
        records, rgb_rows, array_payload,
        history_windows=history_windows,
        context_family="rgb_projection",
        action_frame=action_frame,
    )
    train_mask = np.asarray([str(row["split"]) == "train" for row in records])
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in records])
    outcome_mask = np.asarray([str(row["split"]) == "outcome" for row in records])
    train_raw = _split_data(data, train_mask)
    reports: dict[str, Any] = {}
    for frame in ("raw_xy", "goal_relative"):
        fit_data = copy.deepcopy(data)
        fit_data["task_target"] = _to_task_frame(
            data["task_target"], data["source_xy"], data["goals"], frame,
        ).astype(np.float32)
        fit_data["reference_task_target"] = _to_task_frame(
            data["reference_task_target"], data["source_xy"], data["goals"], frame,
        ).astype(np.float32)
        train_fit = _split_data(fit_data, train_mask)
        objective_reports: dict[str, Any] = {}
        for objective in OBJECTIVES:
            seed_reports: list[dict[str, Any]] = []
            seed_effects: list[np.ndarray] = []
            for seed in SEEDS:
                fit = _fit_one(
                    train_fit, fit_data, seed=seed, epochs=epochs, device=_device(device),
                    objective=objective, rgb_weight=1.0, temporal_weight=0.5,
                    ranking_weight=0.5, ranking_margin=0.005,
                )
                predicted_effect = _from_task_frame(
                    fit["task_prediction"], data["source_xy"], data["goals"], frame,
                )
                seed_effects.append(predicted_effect)
                seed_reports.append({
                    "seed": int(seed),
                    "calibration": _report(data, calibration_mask, predicted_effect),
                    "outcome": _report(data, outcome_mask, predicted_effect),
                    "training": fit["training"],
                })
            stack = np.stack(seed_effects, axis=0)
            mean_effect = stack.mean(axis=0)
            outcome = _report(data, outcome_mask, mean_effect)
            objective_reports[objective] = {
                "seeds": seed_reports,
                "ensemble": {
                    "calibration": _report(data, calibration_mask, mean_effect),
                    "outcome": outcome,
                },
                "root_bootstrap_ci95": _root_bootstrap(
                    outcome["paired_regret_gain_values"],
                ),
            }
        reports[frame] = objective_reports
    result = {
        "schema": "pointmaze_task_frame_intervention.v1",
        "dataset": str(dataset.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "arrays": str(arrays.resolve()),
        "protocol": {
            "fit_split": "train roots only",
            "calibration_split": "reported only; no frame or objective selected",
            "outcome_used_for_selection": False,
            "action_frame": action_frame,
            "task_frames": ["raw_xy", "goal_relative"],
            "goal_relative_definition": "projection onto source-to-goal direct/perpendicular basis",
            "prediction_evaluation_frame": "physical raw XY after inverse transform",
            "device": str(_device(device)),
            "seeds": list(SEEDS),
        },
        "objectives": reports,
        "claim_boundary": "coordinate-frame intervention for an intervention-supervised PointMaze adapter; not frozen RGB online-planning evidence or universal transfer",
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
    parser.add_argument("--action-frame", choices=("raw_control", "goal_relative", "hybrid"), default="hybrid")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({
        frame: {
            objective: value["ensemble"]["outcome"]["paired_regret_gain_mean"]
            for objective, value in objectives.items()
        }
        for frame, objectives in result["objectives"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
