"""Diagnose representation and action-interface conditions in PointMaze.

This is a pre-selection diagnostic.  It uses only the frozen export and the
train/calibration split, never outcome rows.  It is intended to explain why a
support-matched learned-adapter panel can still differ across visual
representations after the action chart is corrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import (
        HORIZONS, _pca, _project, _task_direction,
    )
except ModuleNotFoundError:
    from train_pointmaze_learned_adapter_panel import HORIZONS, _pca, _project, _task_direction


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    centered = x - mean_x
    scale = max(float(np.trace(centered.T @ centered) / max(len(x) * x.shape[1], 1)), 1e-8)
    weights = np.linalg.solve(
        centered.T @ centered / max(len(x), 1) + ridge * scale * np.eye(x.shape[1]),
        centered.T @ (y - mean_y) / max(len(x), 1),
    )
    return weights, mean_x, mean_y


def _predict(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean_x, mean_y = fit
    return (np.asarray(x, dtype=np.float64) - mean_x) @ weights + mean_y


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def _load(path: Path, max_rows: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = sorted(
        payload["enriched_records"],
        key=lambda row: (int(row["episode_id"]), int(row.get("window_start", 0))),
    )
    if len(rows) < max_rows:
        raise ValueError(f"{path} has {len(rows)} rows, fewer than max_rows={max_rows}")
    return rows[:max_rows]


def diagnose(
    paths: dict[str, Path], *, max_rows: int = 32, train_fraction: float = 0.65,
    min_calibration_rows: int = 8,
) -> dict[str, Any]:
    if max_rows < 24:
        raise ValueError("max_rows must leave at least 24 rows for train/calibration/outcome protocol")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if min_calibration_rows < 1:
        raise ValueError("min_calibration_rows must be positive")
    reports = {}
    outcome_count = min(32, max(8, max_rows // 3))
    fit_count = max_rows - outcome_count
    train_count = max(8, int(round(train_fraction * fit_count)))
    train_count = min(train_count, fit_count - min_calibration_rows)
    calibration_count = fit_count - train_count
    if calibration_count < min_calibration_rows:
        raise ValueError(
            f"split leaves only {calibration_count} calibration rows; "
            f"need at least {min_calibration_rows}"
        )
    for base, path in sorted(paths.items()):
        rows = _load(path, max_rows)
        train = rows[:train_count]
        calibration = rows[train_count:fit_count]
        pca_mean, pca_basis, _ = _pca(train, rank=8)
        source_train = np.asarray([_project(row["source_latent"], pca_mean, pca_basis) for row in train])
        source_cal = np.asarray([_project(row["source_latent"], pca_mean, pca_basis) for row in calibration])
        actions_raw_train = np.asarray([np.asarray(row["action_blocks_raw"]).reshape(-1) for row in train])
        actions_raw_cal = np.asarray([np.asarray(row["action_blocks_raw"]).reshape(-1) for row in calibration])
        actions_export_train = np.asarray([np.asarray(row["action_blocks_normalized"]).reshape(-1) for row in train])
        actions_export_cal = np.asarray([np.asarray(row["action_blocks_normalized"]).reshape(-1) for row in calibration])
        target_state_train = {
            h: np.asarray([row["latent_by_horizon"][str(h)]["target_state_raw"] for row in train])
            for h in HORIZONS
        }
        direction = _task_direction(
            np.asarray([row["source_state_raw"] for row in train]),
            target_state_train[1],
        )
        horizon_reports = {}
        for horizon in HORIZONS:
            target_train = np.asarray([
                _project(row["latent_by_horizon"][str(horizon)]["target"], pca_mean, pca_basis)
                for row in train
            ])
            target_cal = np.asarray([
                _project(row["latent_by_horizon"][str(horizon)]["target"], pca_mean, pca_basis)
                for row in calibration
            ])
            delta_train = target_train - source_train
            delta_cal = target_cal - source_cal
            action_fit = _ridge_fit(actions_raw_train, delta_train)
            state_action_fit = _ridge_fit(np.concatenate([source_train, actions_raw_train], axis=1), delta_train)
            horizon_reports[str(horizon)] = {
                "latent_delta_norm_mean": float(np.mean(np.linalg.norm(delta_train, axis=1))),
                "latent_target_variance": float(np.mean(np.sum((target_train - target_train.mean(0)) ** 2, axis=1))),
                "action_only_calibration_r2": _r2(delta_cal, _predict(actions_raw_cal, action_fit)),
                "state_action_calibration_r2": _r2(
                    delta_cal,
                    _predict(np.concatenate([source_cal, actions_raw_cal], axis=1), state_action_fit),
                ),
                "task_displacement_train_std": float(np.std(np.asarray([
                    (np.asarray(row["latent_by_horizon"][str(horizon)]["target_state_raw"])[:2]
                     - np.asarray(row["source_state_raw"])[:2]) @ direction
                    for row in train
                ]))),
            }
        singular_values = np.linalg.svd(actions_raw_train - actions_raw_train.mean(0), compute_uv=False)
        reports[base] = {
            "source": str(path.resolve()),
            "rows": max_rows,
            "train_rows": train_count,
            "calibration_rows": len(calibration),
            "outcome_rows": outcome_count,
            "action_interface": {
                "raw_mean": float(actions_raw_train.mean()),
                "raw_std": float(actions_raw_train.std()),
                "raw_min": float(actions_raw_train.min()),
                "raw_max": float(actions_raw_train.max()),
                "export_field_mean": float(actions_export_train.mean()),
                "export_field_std": float(actions_export_train.std()),
                "export_field_min": float(actions_export_train.min()),
                "export_field_max": float(actions_export_train.max()),
                "raw_calibration_std": float(actions_raw_cal.std()),
                "effective_rank": int(np.sum(singular_values > singular_values[0] * 1e-6)),
                "action_feature_used_by_panel": "raw action_blocks_raw",
            },
            "horizons": horizon_reports,
        }
    return {
        "schema": "pointmaze_representation_condition_diagnostic.v1",
        "protocol": {
            "support_matching": True,
            "max_rows": max_rows,
            "train_fraction": train_fraction,
            "min_calibration_rows": min_calibration_rows,
            "action_input": "raw shared physical action blocks",
            "fit_split": "train rows only",
            "calibration_split": "calibration rows only",
            "outcome_used_for_selection": False,
            "task_direction": "train-only h=1 source-to-target displacement",
        },
        "representations": reports,
        "interpretation": (
            "A representation is action-transfer sufficient only if the shared raw action chart, "
            "held-out action-to-latent effect, and task readout remain stable. High calibration "
            "error after raw-action correction is evidence of representation-conditioned transfer "
            "failure, not permission to tune on outcome rows."
        ),
        "claim_boundary": "Diagnostic mechanism evidence only; no cross-environment or universal-world-model claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-export", type=Path, required=True)
    parser.add_argument("--dino-export", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--min-calibration-rows", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        {"jepa": args.jepa_export, "dino": args.dino_export},
        max_rows=args.max_rows,
        train_fraction=args.train_fraction,
        min_calibration_rows=args.min_calibration_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": report["schema"], "representations": report["representations"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
