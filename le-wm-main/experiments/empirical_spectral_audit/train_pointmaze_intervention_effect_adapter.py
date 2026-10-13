"""Train an intervention-supervised PointMaze task-effect adapter.

This is a deliberately separate workstream from the observed-transition
matched benchmark.  The latter asks whether geometry has incremental value
after matching prediction quality; it cannot create counterfactual action
information that was absent from the training trajectories.  This runner
tests that mechanism directly:

* source latents remain frozen RGB-derived features;
* train/calibration/evaluation roots are disjoint by ``window_start``;
* simulator state is used only offline to generate intervention labels;
* the model is trained on balanced ``+a/-a/zero`` effects;
* evaluation reports effect risk and action-ranking regret on fresh roots.

An improvement here is evidence for intervention supervision, not evidence
for a universal world model or an independent geometry selector.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _balanced_action_risk,
    _intervention_action_blocks,
    _simulate_effects,
)
from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import (
    HORIZONS,
    _pca,
    _project,
)


SEEDS = (8411, 8412, 8413, 8414, 8415)
ACTION_MAGNITUDE = 0.75


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _directions() -> tuple[np.ndarray, ...]:
    return (
        np.asarray([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
        np.asarray([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0),
    )


def _fan_actions(direction: np.ndarray) -> list[np.ndarray]:
    return [
        ACTION_MAGNITUDE * direction,
        -ACTION_MAGNITUDE * direction,
        np.zeros(2, dtype=np.float64),
    ]


def _rows_by_window(payload: dict[str, Any], window_start: int) -> list[dict[str, Any]]:
    rows = [
        row for row in payload["enriched_records"]
        if int(row.get("window_start", 0)) == int(window_start)
    ]
    if len(rows) < 8:
        raise ValueError(f"window_start={window_start} has too few rows: {len(rows)}")
    return sorted(rows, key=lambda row: (int(row["episode_id"]), int(row["window_start"])))


class EffectMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.SiLU(),
            nn.Linear(96, 96), nn.SiLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _make_samples(
    rows: list[dict[str, Any]],
    pca_mean: np.ndarray,
    pca_basis: np.ndarray,
    previous_rows: dict[int, dict[str, Any]] | None = None,
    feature_source: str = "latent",
) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for direction in _directions():
        fan_actions = _fan_actions(direction)
        actual = _simulate_effects(rows, fan_actions)  # [N, fan, horizon, xy]
        for row_index, row in enumerate(rows):
            if feature_source == "privileged_state":
                source = np.asarray(row["source_state_raw"], dtype=np.float64)
            else:
                source = _project(row["source_latent"], pca_mean, pca_basis)
            if previous_rows is not None:
                previous = previous_rows.get(int(row["episode_id"]))
                if previous is None:
                    raise ValueError(f"missing previous latent for episode {row['episode_id']}")
                previous_source = (
                    np.asarray(previous["source_state_raw"], dtype=np.float64)
                    if feature_source == "privileged_state"
                    else _project(previous["source_latent"], pca_mean, pca_basis)
                )
                source = np.concatenate([source, source - previous_source])
            for fan_index, action in enumerate(fan_actions):
                blocks = _intervention_action_blocks(action).reshape(-1)
                features.append(np.concatenate([source, blocks]))
                targets.append(actual[row_index, fan_index].reshape(-1))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed(seed)
    x_mean = train_x.mean(axis=0)
    x_scale = np.maximum(train_x.std(axis=0), 1e-3)
    y_mean = train_y.mean(axis=0)
    y_scale = np.maximum(train_y.std(axis=0), 1e-3)
    x = torch.as_tensor((train_x - x_mean) / x_scale, dtype=torch.float32)
    y = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32)
    model = EffectMLP(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        prediction = model(torch.as_tensor((eval_x - x_mean) / x_scale, dtype=torch.float32)).numpy()
    prediction = prediction * y_scale + y_mean
    return prediction, {
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "epochs": int(epochs),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _evaluate_effects(
    prediction: np.ndarray,
    actual: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    # Samples are ordered direction, row, fan.  Restore [N, direction, fan,
    # horizon, xy] before computing the balanced intervention estimand.
    n = len(rows)
    prediction = prediction.reshape(2, n, 3, len(HORIZONS), 2).transpose(1, 0, 2, 3, 4)
    actual = actual.reshape(2, n, 3, len(HORIZONS), 2).transpose(1, 0, 2, 3, 4)
    horizon_risk, horizon_identifiability = {}, {}
    for index, horizon in enumerate(HORIZONS):
        risks, identifiability = [], []
        for direction_index in range(len(_directions())):
            risk, ident, _ = _balanced_action_risk(
                prediction[:, direction_index, index], actual[:, direction_index, index]
            )
            risks.append(risk)
            identifiability.append(ident)
        horizon_risk[str(horizon)] = float(np.mean(risks))
        horizon_identifiability[str(horizon)] = float(np.mean(identifiability))

    ranking_regret = {}
    for index, horizon in enumerate(HORIZONS):
        # Choose among the four nonzero diagonal pulses using predicted
        # x+y progress, then score that choice using the offline physical
        # effect.  This is an independent action-ranking consequence.
        pred_nonzero = prediction[:, :, :2, index, :].reshape(n, 4, 2)
        actual_nonzero = actual[:, :, :2, index, :].reshape(n, 4, 2)
        selected = np.argmax(pred_nonzero.sum(axis=-1), axis=1)
        chosen = actual_nonzero[np.arange(n), selected].sum(axis=-1)
        oracle = np.max(actual_nonzero.sum(axis=-1), axis=1)
        ranking_regret[str(horizon)] = {
            "mean_regret": float(np.mean(oracle - chosen)),
            "mean_chosen_progress": float(np.mean(chosen)),
            "mean_oracle_progress": float(np.mean(oracle)),
        }
    return {
        "action_inverse_risk": float(np.mean(list(horizon_risk.values()))),
        "horizon_action_inverse_risk": horizon_risk,
        "horizon_action_identifiability": horizon_identifiability,
        "action_ranking": ranking_regret,
        "rows": int(n),
    }


def run(
    export: Path,
    output: Path,
    *,
    train_window: int = 5,
    calibration_window: int = 10,
    evaluation_window: int = 15,
    epochs: int = 300,
    use_history: bool = False,
    feature_source: str = "latent",
) -> dict[str, Any]:
    if feature_source not in {"latent", "privileged_state"}:
        raise ValueError("feature_source must be latent or privileged_state")
    if feature_source == "privileged_state" and use_history:
        raise ValueError("privileged_state and history are separate diagnostic conditions")
    payload = load_payload(export)
    train_rows = _rows_by_window(payload, train_window)
    calibration_rows = _rows_by_window(payload, calibration_window)
    evaluation_rows = _rows_by_window(payload, evaluation_window)
    previous_rows_by_window = {
        window: {
            int(row["episode_id"]): row
            for row in payload["enriched_records"]
            if int(row.get("window_start", 0)) == window
        }
        for window in (train_window - 5, calibration_window - 5, evaluation_window - 5)
    }
    if use_history and any(not rows for rows in previous_rows_by_window.values()):
        raise ValueError("history mode requires the preceding five-step window for every split")
    pca_mean, pca_basis, _ = _pca(train_rows, rank=8)
    train_x, train_y = _make_samples(
        train_rows, pca_mean, pca_basis,
        previous_rows_by_window[train_window - 5] if use_history else None,
        feature_source,
    )
    calibration_x, calibration_y = _make_samples(
        calibration_rows, pca_mean, pca_basis,
        previous_rows_by_window[calibration_window - 5] if use_history else None,
        feature_source,
    )
    evaluation_x, evaluation_y = _make_samples(
        evaluation_rows, pca_mean, pca_basis,
        previous_rows_by_window[evaluation_window - 5] if use_history else None,
        feature_source,
    )
    reports = []
    for seed in SEEDS:
        calibration_prediction, training = _fit_predict(train_x, train_y, calibration_x, seed, epochs)
        evaluation_prediction, _ = _fit_predict(train_x, train_y, evaluation_x, seed, epochs)
        calibration = _evaluate_effects(calibration_prediction, calibration_y, calibration_rows)
        evaluation = _evaluate_effects(evaluation_prediction, evaluation_y, evaluation_rows)
        reports.append({"seed": int(seed), "calibration": calibration, "evaluation": evaluation, "training": training})
    report = {
        "schema": "pointmaze_intervention_effect_adapter.v1",
        "source_export": str(export.resolve()),
        "protocol": {
            "train_window_start": int(train_window),
            "calibration_window_start": int(calibration_window),
            "evaluation_window_start": int(evaluation_window),
            "root_disjoint": True,
            "source_latent": "frozen RGB-derived source latent projected by train-only PCA" if feature_source == "latent" else "privileged simulator source state upper bound",
            "intervention": "+a/-a/zero diagonal fan repeated over four five-step blocks",
            "labels": "offline modern-mujoco replay of released legacy PointMaze XML",
            "runtime_state_access": False,
            "model_target": "physical task effect, not latent successor",
            "history_features": "current projected latent plus z_t-z_(t-5)" if use_history else "current projected latent only",
            "history_previous_window": 5,
            "feature_source": feature_source,
            "outcome_used_for_training_or_selection": False,
        },
        "reports": reports,
        "summary": {
            "evaluation_action_inverse_risk_mean": float(np.mean([r["evaluation"]["action_inverse_risk"] for r in reports])),
            "evaluation_action_inverse_risk_std": float(np.std([r["evaluation"]["action_inverse_risk"] for r in reports])),
            "evaluation_horizon_risk_mean": {
                str(h): float(np.mean([r["evaluation"]["horizon_action_inverse_risk"][str(h)] for r in reports]))
                for h in HORIZONS
            },
            "evaluation_ranking_regret_mean": {
                str(h): float(np.mean([r["evaluation"]["action_ranking"][str(h)]["mean_regret"] for r in reports]))
                for h in HORIZONS
            },
        },
        "claim_boundary": "Intervention-supervised PointMaze task-effect adapter; not evidence for a geometry selector or universal world model.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-window", type=int, default=5)
    parser.add_argument("--calibration-window", type=int, default=10)
    parser.add_argument("--evaluation-window", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use-history", action="store_true")
    parser.add_argument("--feature-source", choices=("latent", "privileged_state"), default="latent")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
