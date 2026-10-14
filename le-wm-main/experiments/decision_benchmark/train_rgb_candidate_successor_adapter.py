"""Train a train-root-only RGB candidate-successor adapter.

The frozen LeWM predictor is trained on observed trajectories, while the
decision benchmark evaluates counterfactual candidate action sequences.  This
runner tests that protocol mismatch directly: it fits a small residual adapter
from source RGB projection plus the complete candidate action sequence to the
realized RGB successor projection, using train roots only.  Calibration and
outcome roots are never used for fitting, epoch selection, or normalization.

This is an intervention-supervised adapter experiment.  It is not evidence
that the frozen LeWM predictor itself has acquired counterfactual dynamics or
that RGB online planning is solved.
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


SEEDS = (8411, 8412, 8413, 8414, 8415)


class CandidateResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing silent CPU fallback")
        return torch.device("cuda")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def _load_rows(dataset: Path, rgb_report: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    dataset_payload = json.loads(dataset.read_text(encoding="utf-8"))
    rgb_payload = json.loads(rgb_report.read_text(encoding="utf-8"))
    records = list(dataset_payload["records"])
    rgb_rows = {
        (str(row["instance_id"]), int(row["candidate_id"])): row
        for row in rgb_payload["rows"]
    }
    missing = [
        (str(row["instance_id"]), int(row["candidate_id"]))
        for row in records
        if (str(row["instance_id"]), int(row["candidate_id"])) not in rgb_rows
    ]
    if missing:
        raise ValueError(f"RGB report lacks {len(missing)} candidate rows")
    return records, rgb_rows


def _features(
    records: list[dict[str, Any]],
    rgb_rows: dict[tuple[str, int], dict[str, Any]],
    *,
    history_windows: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if history_windows < 1:
        raise ValueError("history_windows must be positive")
    features = []
    targets = []
    sources = []
    for record in records:
        rgb = rgb_rows[(str(record["instance_id"]), int(record["candidate_id"]))]
        source = np.asarray(rgb["source_projection"], dtype=np.float64)
        if history_windows == 1:
            context = source
        else:
            if "source_history_projection" not in rgb:
                raise ValueError("RGB report lacks source_history_projection for history adapter")
            history = np.asarray(rgb["source_history_projection"], dtype=np.float64)
            if history.ndim != 2 or history.shape[0] != history_windows:
                raise ValueError(
                    "source history shape does not match requested history_windows: "
                    f"{history.shape} vs {history_windows}"
                )
            context = history.reshape(-1)
        action = np.asarray(record["action_sequence"], dtype=np.float64).reshape(-1)
        target = np.asarray(rgb["realized_terminal_projection"], dtype=np.float64)
        features.append(np.concatenate([context, action]))
        targets.append(target - source)
        sources.append(source)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(sources, dtype=np.float32)


def _fit_one(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_all: np.ndarray,
    source_all: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed(seed)
    x_mean = x_train.mean(axis=0)
    x_scale = np.maximum(x_train.std(axis=0), 1.0e-3)
    y_mean = residual_train.mean(axis=0)
    y_scale = np.maximum(residual_train.std(axis=0), 1.0e-3)
    x_fit = torch.as_tensor((x_train - x_mean) / x_scale, dtype=torch.float32, device=device)
    y_fit = torch.as_tensor((residual_train - y_mean) / y_scale, dtype=torch.float32, device=device)
    model = CandidateResidualMLP(x_train.shape[1], residual_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_fit)
        loss = torch.mean((prediction - y_fit) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        x_all_tensor = torch.as_tensor((x_all - x_mean) / x_scale, dtype=torch.float32, device=device)
        residual = model(x_all_tensor).cpu().numpy() * y_scale + y_mean
    prediction = source_all + residual
    return prediction.astype(np.float64), {
        "seed": int(seed),
        "epochs": int(epochs),
        "final_train_loss": float(history[-1]),
    }


def run(
    dataset: Path,
    rgb_report: Path,
    output: Path,
    *,
    epochs: int = 250,
    device: str = "cuda",
    history_windows: int = 1,
) -> dict[str, Any]:
    runtime_device = _device(device)
    records, rgb_rows = _load_rows(dataset, rgb_report)
    x_all, residual_all, source_all = _features(records, rgb_rows, history_windows=history_windows)
    train_mask = np.asarray([str(row["split"]) == "train" for row in records])
    if int(np.sum(train_mask)) < 2:
        raise ValueError("at least two train-root candidate rows are required")
    predictions = []
    training_reports = []
    for seed in SEEDS:
        prediction, report = _fit_one(
            x_all[train_mask], residual_all[train_mask], x_all, source_all,
            seed=seed, epochs=epochs, device=runtime_device,
        )
        predictions.append(prediction)
        training_reports.append(report)
    ensemble_prediction = np.mean(np.stack(predictions), axis=0)
    output_rows = []
    for record, prediction in zip(records, ensemble_prediction):
        rgb = rgb_rows[(str(record["instance_id"]), int(record["candidate_id"]))]
        realized = np.asarray(rgb["realized_terminal_projection"], dtype=np.float64)
        error = prediction - realized
        output_rows.append({
            "instance_id": str(record["instance_id"]),
            "candidate_id": int(record["candidate_id"]),
            "split": str(record["split"]),
            "source_episode_id": int(record["start_episode_id"]),
            "renderer_consistency_pass": bool(rgb["renderer_consistency_pass"]),
            "source_latent_projection_l1": float(rgb["source_latent_projection_l1"]),
            "source_projection": np.asarray(rgb["source_projection"], dtype=np.float64).tolist(),
            "predicted_terminal_projection": prediction.tolist(),
            "realized_terminal_projection": realized.tolist(),
            "predicted_vs_realized_projection_l1": float(np.mean(np.abs(error))),
            "predicted_vs_realized_projection_rmse": float(np.sqrt(np.mean(error**2))),
            "predicted_vs_realized_projection_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(realized), 1.0e-8)),
            "realized_terminal_state": rgb["realized_terminal_state"],
            "realized_terminal_rgb_mean": float(rgb["realized_terminal_rgb_mean"]),
            "realized_terminal_raw_latent": rgb["realized_terminal_raw_latent"],
            "prediction_projection_status": "train_root_only_rgb_candidate_successor_adapter_ensemble",
        })

    def _summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"rows": 0}
        errors = np.asarray([row["predicted_vs_realized_projection_rmse"] for row in selected], dtype=np.float64)
        return {"rows": int(len(selected)), "projection_rmse_mean": float(np.mean(errors)), "projection_rmse_median": float(np.median(errors))}

    by_split = {
        split: _summary([row for row in output_rows if row["split"] == split])
        for split in ("train", "calibration", "outcome")
    }
    report = {
        "schema": "pointmaze_rgb_candidate_successor_adapter.v1",
        "dataset": str(dataset.resolve()),
        "rgb_report": str(rgb_report.resolve()),
        "record_count": len(output_rows),
        "rows": output_rows,
        "summary": {"all": _summary(output_rows), "by_split": by_split},
        "training": {
            "fit_split": "train roots only",
            "normalization_fit_split": "train roots only",
            "epoch_selection": "fixed predeclared epoch count; external calibration and outcome are not used",
            "epochs": int(epochs),
            "seeds": [int(seed) for seed in SEEDS],
            "ensemble": "arithmetic mean of five train-root-only residual MLPs",
            "input": "source RGB PCA projection plus complete candidate action sequence",
            "history_windows": int(history_windows),
            "history_context": "causal source RGB projections flattened oldest-to-current; current source remains the residual origin",
            "target": "realized RGB terminal projection minus source RGB projection",
            "device": str(runtime_device),
        },
        "gates": {
            "train_root_only_fit": True,
            "realized_rgb_successor_available": bool(output_rows),
            "closed_loop_online_planning": False,
        },
        "claim_boundary": "intervention-supervised RGB candidate-successor adapter; not frozen LeWM counterfactual dynamics, online planning, or universal transfer",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--history-windows", type=int, default=1)
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
