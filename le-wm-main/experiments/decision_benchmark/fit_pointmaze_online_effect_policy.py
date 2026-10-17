"""Freeze a leak-free PointMaze action-effect ensemble for online RGB use.

Only train-root intervention labels fit the five effect models.  Calibration
roots determine the finite-sample root-regret bound.  Outcome roots are not
read by this script.  The artifact also stores the train-only latent PCA and
nearest-support models needed to fail closed at runtime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.task_contrast import calibrate_root_score_bound
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    EffectMLP,
    _features,
    _root_selection,
    action_sequence_basis,
    fit_effect_model,
    predict_effect_model,
)


SCHEMA = "pointmaze_online_effect_policy.v1"


def _nearest_distances(values: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(
        (values[:, None, :] - references[None, :, :]) ** 2, axis=-1,
    ))


def fit_nearest_support(values: np.ndarray, *, quantile: float = 0.9) -> dict[str, Any]:
    """Fit a train-only standardized nearest-neighbour support envelope."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("support fitting requires at least three finite vectors")
    unique = np.unique(array, axis=0)
    if len(unique) < 3:
        raise ValueError("support fitting requires at least three unique vectors")
    mean = unique.mean(axis=0)
    scale = np.maximum(unique.std(axis=0), 1.0e-6)
    standardized = (unique - mean) / scale
    distances = _nearest_distances(standardized, standardized)
    np.fill_diagonal(distances, np.inf)
    leave_one_out = distances.min(axis=1)
    threshold = float(np.quantile(leave_one_out, quantile, method="higher"))
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "references": standardized.tolist(),
        "threshold": threshold,
        "quantile": float(quantile),
        "reference_count": int(len(unique)),
        "leave_one_out_distances": leave_one_out.tolist(),
    }


def support_distance(model: dict[str, Any], value: np.ndarray) -> float:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    references = np.asarray(model["references"], dtype=np.float64)
    if vector.shape != mean.shape or references.shape[1] != len(vector):
        raise ValueError("support query dimension does not match the frozen model")
    query = (vector - mean) / scale
    return float(_nearest_distances(query[None], references).min())


@dataclass
class OnlineEffectPolicy:
    models: list[EffectMLP]
    normalizations: list[dict[str, Any]]
    pca_mean: np.ndarray
    pca_basis: np.ndarray
    latent_support: dict[str, Any]
    action_support: dict[str, Any]
    regret_calibration: dict[str, Any]
    action_steps: int
    feature_mode: str
    history_windows: int
    device: torch.device

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "OnlineEffectPolicy":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"unsupported online effect-policy schema: {payload.get('schema')}")
        models: list[EffectMLP] = []
        normalizations: list[dict[str, Any]] = []
        for member in payload["ensemble"]:
            metadata = dict(member["normalization"])
            model = EffectMLP(int(metadata["input_dim"]))
            model.load_state_dict(member["state_dict"])
            model.to(device).eval()
            models.append(model)
            normalizations.append(metadata)
        if not models:
            raise ValueError("effect-policy ensemble cannot be empty")
        return cls(
            models=models,
            normalizations=normalizations,
            pca_mean=np.asarray(payload["pca_mean"], dtype=np.float64),
            pca_basis=np.asarray(payload["pca_basis"], dtype=np.float64),
            latent_support=dict(payload["latent_support"]),
            action_support=dict(payload["action_support"]),
            regret_calibration=dict(payload["regret_calibration"]),
            action_steps=int(payload["action_steps"]),
            feature_mode=str(payload.get("feature_mode", "frozen_latent_action_basis")),
            history_windows=int(payload.get("history_windows", 1)),
            device=device,
        )

    def project_latent(self, pooled_latent: np.ndarray) -> np.ndarray:
        value = np.asarray(pooled_latent, dtype=np.float64).reshape(-1)
        if value.shape != self.pca_mean.shape or self.pca_basis.shape[1] != len(value):
            raise ValueError("pooled latent does not match the frozen PCA chart")
        return (value - self.pca_mean) @ self.pca_basis.T

    def project_history(self, pooled_latents: np.ndarray) -> np.ndarray:
        values = np.asarray(pooled_latents, dtype=np.float64)
        if values.ndim != 2 or len(values) < self.history_windows:
            raise ValueError("insufficient pooled latent history for effect policy")
        selected = values[-self.history_windows :]
        projected = np.stack([self.project_latent(value) for value in selected])
        return projected.reshape(-1)

    def score(
        self,
        source_projection: np.ndarray,
        current_xy: np.ndarray,
        target_xy: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, Any]:
        projection = np.asarray(source_projection, dtype=np.float64).reshape(-1)
        candidates = np.asarray(actions, dtype=np.float64)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("online effect-policy actions must have shape [N, 2]")
        bases = np.stack([
            action_sequence_basis(np.repeat(action[None], self.action_steps, axis=0))
            for action in candidates
        ])
        features = np.concatenate([
            np.repeat(projection[None], len(candidates), axis=0), bases,
        ], axis=1).astype(np.float32)
        members = [
            predict_effect_model(model, metadata, features)
            for model, metadata in zip(self.models, self.normalizations)
        ]
        effects = np.mean(np.stack(members), axis=0)
        terminals = np.asarray(current_xy, dtype=np.float64).reshape(1, 2) + effects
        costs = np.linalg.norm(
            terminals - np.asarray(target_xy, dtype=np.float64).reshape(1, 2), axis=1,
        )
        action_distances = np.asarray([
            support_distance(self.action_support, basis) for basis in bases
        ])
        return {
            "costs": costs,
            "effects": effects,
            "terminals": terminals,
            "action_support_distances": action_distances,
            "latent_support_distance": support_distance(self.latent_support, projection),
        }


def build(
    candidate_dataset: Path,
    arrays: Path,
    source_model_checkpoint: Path,
    output: Path,
    summary_output: Path,
    *,
    calibration_candidate_dataset: Path | None = None,
    calibration_arrays: Path | None = None,
    epochs: int = 300,
    alpha: float = 0.1,
    device: str = "cuda",
    feature_mode: str = "frozen_latent_action_basis",
    calibration_splits: list[str] | tuple[str, ...] = ("calibration",),
) -> dict[str, Any]:
    payload = json.loads(candidate_dataset.read_text(encoding="utf-8"))
    records = payload["records"]
    arrays_payload = np.load(arrays, allow_pickle=False)
    if feature_mode not in {"frozen_latent_action_basis", "history_latent_action_basis"}:
        raise ValueError("online effect policy requires latent-only deployment features")
    features, effects, _roots, _goals = _features(records, arrays_payload, feature_mode)
    train_mask = np.asarray([row.get("split") == "train" for row in records])
    if (calibration_candidate_dataset is None) != (calibration_arrays is None):
        raise ValueError(
            "calibration_candidate_dataset and calibration_arrays must be provided together"
        )
    calibration_dataset_path = calibration_candidate_dataset or candidate_dataset
    calibration_arrays_path = calibration_arrays or arrays
    calibration_payload = json.loads(
        calibration_dataset_path.read_text(encoding="utf-8")
    )
    calibration_all_records = calibration_payload["records"]
    calibration_array_payload = np.load(calibration_arrays_path, allow_pickle=False)
    calibration_features, _calibration_effects, _calibration_roots, calibration_goals = _features(
        calibration_all_records, calibration_array_payload, feature_mode,
    )
    calibration_split_set = {str(value) for value in calibration_splits}
    if not calibration_split_set or "train" in calibration_split_set:
        raise ValueError("calibration_splits must be non-empty and cannot include train")
    calibration_mask = np.asarray([
        str(row.get("split")) in calibration_split_set
        for row in calibration_all_records
    ])
    if not train_mask.any() or not calibration_mask.any():
        raise ValueError("candidate bank requires train and calibration roots")

    source_payload = torch.load(source_model_checkpoint, map_location="cpu", weights_only=False)
    pca_mean = np.asarray(source_payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(source_payload["pca_basis"], dtype=np.float64)
    latent_dim = int(pca_basis.shape[0])
    history_windows = int((features.shape[1] - 16) // latent_dim)
    if history_windows < 1 or features.shape[1] != history_windows * latent_dim + 16:
        raise ValueError("candidate features do not match source-model PCA plus action basis")
    if feature_mode == "frozen_latent_action_basis" and history_windows != 1:
        raise ValueError("single-latent feature mode unexpectedly contains history")

    ensemble = []
    predictions = []
    for seed in SEEDS:
        model, metadata = fit_effect_model(
            features[train_mask], effects[train_mask], seed=seed,
            epochs=epochs, device_name=device,
        )
        predictions.append(predict_effect_model(model, metadata, calibration_features))
        ensemble.append({
            "seed": int(seed),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "normalization": metadata,
        })
    mean_effect = np.mean(np.stack(predictions), axis=0)

    by_train_root: dict[str, np.ndarray] = {}
    for row, keep in zip(records, train_mask):
        if keep:
            by_train_root.setdefault(
                str(row["instance_id"]),
                np.asarray(
                    row[
                        "source_history_model"
                        if feature_mode == "history_latent_action_basis"
                        else "source_latent_model"
                    ],
                    dtype=np.float64,
                ),
            )
    latent_support = fit_nearest_support(np.stack(list(by_train_root.values())))
    calibration_train_mask = np.asarray([
        row.get("split") == "train" for row in calibration_all_records
    ])
    if not calibration_train_mask.any():
        raise ValueError("calibration candidate bank requires train roots for action support")
    train_action_bases = np.stack([
        action_sequence_basis(np.asarray(row["action_sequence"], dtype=np.float64))
        for row, keep in zip(calibration_all_records, calibration_train_mask) if keep
    ])
    action_support = fit_nearest_support(train_action_bases)

    sources = np.asarray([
        row["source_state_raw"] for row in calibration_all_records
    ], dtype=np.float64)[:, :2]
    predicted_cost = np.linalg.norm(sources + mean_effect - calibration_goals, axis=1)
    calibration_records = [
        row for row, keep in zip(calibration_all_records, calibration_mask) if keep
    ]
    calibration_selection = _root_selection(
        calibration_records, predicted_cost[calibration_mask],
    )
    supported_calibration = []
    for root, row in sorted(calibration_selection["roots"].items()):
        source = next(
            np.asarray(
                record[
                    "source_history_model"
                    if feature_mode == "history_latent_action_basis"
                    else "source_latent_model"
                ],
                dtype=np.float64,
            )
            for record in calibration_all_records if str(record["instance_id"]) == root
        )
        if support_distance(latent_support, source) <= float(latent_support["threshold"]):
            supported_calibration.append(float(row["regret"]))
    regret_calibration = calibrate_root_score_bound(
        np.asarray(supported_calibration), alpha=alpha,
    )

    artifact = {
        "schema": SCHEMA,
        "ensemble": ensemble,
        "pca_mean": pca_mean.tolist(),
        "pca_basis": pca_basis.tolist(),
        "latent_support": latent_support,
        "action_support": action_support,
        "regret_calibration": regret_calibration,
        "action_steps": 20,
        "feature_mode": feature_mode,
        "history_windows": history_windows,
        "protocol": {
            "feature_mode": feature_mode,
            "history_windows": history_windows,
            "adapter_seeds": list(SEEDS),
            "epochs": int(epochs),
            "device": device,
            "training_split": "train roots only",
            "calibration_split": "declared development calibration pool only",
            "calibration_splits": sorted(calibration_split_set),
            "development_outcome_reused_for_calibration": "outcome" in calibration_split_set,
            "deployment_test_outcomes_read": False,
            "source_model_checkpoint": str(source_model_checkpoint.resolve()),
            "training_candidate_dataset": str(candidate_dataset.resolve()),
            "calibration_candidate_dataset": str(calibration_dataset_path.resolve()),
            "candidate_family_matched": candidate_dataset.resolve() == calibration_dataset_path.resolve(),
            "online_candidate_boundary": (
                "runtime repeated-pulse candidates require both latent and action-basis "
                "support; calibration is fixed-bank marginal and does not certify adaptive MPC"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    summary = {
        "schema": "pointmaze_online_effect_policy_summary.v1",
        "artifact": str(output.resolve()),
        "train_root_count": int(len(by_train_root)),
        "calibration_root_count": int(len(calibration_selection["roots"])),
        "supported_calibration_root_count": int(len(supported_calibration)),
        "ensemble_size": int(len(ensemble)),
        "latent_rank": latent_dim,
        "history_windows": history_windows,
        "feature_mode": feature_mode,
        "latent_support_threshold": float(latent_support["threshold"]),
        "action_support_threshold": float(action_support["threshold"]),
        "regret_calibration": regret_calibration,
        "claim_boundary": (
            "deployable train/calibration-only repair artifact; online consequence remains "
            "exploratory until evaluated on unopened roots or an independent environment"
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--source-model-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-candidate-dataset", type=Path, default=None)
    parser.add_argument("--calibration-arrays", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--feature-mode",
        choices=("frozen_latent_action_basis", "history_latent_action_basis"),
        default="frozen_latent_action_basis",
    )
    parser.add_argument(
        "--calibration-splits",
        nargs="+",
        choices=("calibration", "outcome"),
        default=["calibration"],
        help=(
            "development splits used only for deployment calibration; adding outcome is "
            "exploratory refreezing and requires a new independent evaluation panel"
        ),
    )
    args = parser.parse_args()
    print(
        "[PointMaze effect policy] fitting five train-root intervention models; "
        "calibration labels are used only for the finite-sample regret bound"
    )
    summary = build(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
