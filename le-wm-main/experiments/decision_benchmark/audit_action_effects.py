"""Audit action identifiability on root-disjoint PointMaze interventions.

This is deliberately separate from prediction MSE and from the trajectory
calibration split.  For each held-out RGB-derived source root it evaluates a
balanced ``+a/-a/zero`` intervention fan, estimates a finite-difference
action-to-task Jacobian, and reports rank, conditioning, identifiable
directions, bootstrap uncertainty, and support coverage.

The simulator is used only to produce offline intervention labels.  Its state
never enters the adapter input or the reported runtime protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _external_features,
    _fit_readouts,
    _intervention_action_blocks,
    _load_adapter,
    _root_key,
    _simulate_effects,
)
from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import HORIZONS, _predict


def _bootstrap_mean(values: np.ndarray, groups: list[str], seed: int, samples: int = 1000) -> dict[str, float]:
    """Root-level percentile CI; rows from one root are never split."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.asarray(sorted(set(groups.tolist())))
    if len(unique) < 2:
        mean = float(np.mean(values))
        return {"estimate": mean, "lower": mean, "upper": mean, "groups": int(len(unique))}
    rng = np.random.default_rng(seed)
    group_values = [values[groups == group] for group in unique]
    means = np.asarray([
        np.mean(np.concatenate([group_values[index] for index in rng.integers(0, len(unique), len(unique))]))
        for _ in range(samples)
    ])
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "groups": int(len(unique)),
    }


def _matrix_summary(matrices: np.ndarray, threshold: float = 1e-3) -> dict[str, Any]:
    """Summarize a batch of output-by-action Jacobians."""
    matrices = np.asarray(matrices, dtype=np.float64)
    mean_matrix = np.mean(matrices, axis=0)
    singular_values = np.linalg.svd(mean_matrix, compute_uv=False)
    scale = max(float(singular_values[0]) if len(singular_values) else 0.0, 1e-12)
    rank = int(np.sum(singular_values > threshold * scale))
    condition = None
    if len(singular_values) and singular_values[-1] > threshold * scale:
        condition = float(singular_values[0] / singular_values[-1])
    per_root_rank = []
    per_root_condition = []
    for matrix in matrices:
        values = np.linalg.svd(matrix, compute_uv=False)
        local_scale = max(float(values[0]) if len(values) else 0.0, 1e-12)
        per_root_rank.append(int(np.sum(values > threshold * local_scale)))
        per_root_condition.append(
            float(values[0] / values[-1]) if len(values) and values[-1] > threshold * local_scale else None
        )
    _, _, vh = np.linalg.svd(mean_matrix, full_matrices=False)
    identifiable = vh[:rank].tolist() if rank else []
    finite_conditions = [value for value in per_root_condition if value is not None]
    return {
        "mean_jacobian": mean_matrix.tolist(),
        "singular_values": singular_values.tolist(),
        "rank": rank,
        "condition_number": condition,
        "identifiable_input_directions": identifiable,
        "per_root_rank_mean": float(np.mean(per_root_rank)) if per_root_rank else None,
        "per_root_full_rank_fraction": float(np.mean(np.asarray(per_root_rank) == min(matrices.shape[-2:]))) if per_root_rank else None,
        "per_root_condition_median": float(np.median(finite_conditions)) if finite_conditions else None,
        "rank_threshold_relative": threshold,
    }


def audit(
    checkpoint: Path,
    fit_export: Path,
    outcome_export: Path,
    external_export: Path,
    output: Path,
    *,
    device: str = "cpu",
    action_magnitude: float = 0.75,
    max_roots: int | None = None,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    fit = load_payload(fit_export)
    outcome = load_payload(outcome_export)
    external = load_payload(external_export)
    if str(Path(fit["checkpoint"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("fit export checkpoint does not match checkpoint")
    if str(Path(outcome["checkpoint"]).resolve()) != str(checkpoint.resolve()):
        raise ValueError("outcome export checkpoint does not match checkpoint")
    used = {_root_key(row) for row in fit["enriched_records"]} | {_root_key(row) for row in outcome["enriched_records"]}
    rows = [
        row for row in external["enriched_records"]
        if int(row.get("window_start", 0)) != 0 and _root_key(row) not in used
    ]
    if max_roots is not None:
        rows = rows[: int(max_roots)]
    if len(rows) < 4:
        raise ValueError("external export has fewer than four root-disjoint source roots")

    model, payload = _load_adapter(checkpoint, device)
    action_feature_chart = str(payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
    pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(payload["pca_basis"], dtype=np.float64)
    readouts = _fit_readouts(fit["enriched_records"])
    zero = np.zeros(2, dtype=np.float64)
    axis_actions = [
        action_magnitude * np.asarray([1.0, 0.0]),
        -action_magnitude * np.asarray([1.0, 0.0]),
        action_magnitude * np.asarray([0.0, 1.0]),
        -action_magnitude * np.asarray([0.0, 1.0]),
        zero,
    ]
    predicted_by_action: list[dict[int, np.ndarray]] = []
    for action in axis_actions:
        blocks = _intervention_action_blocks(action)
        x = _external_features(
            rows,
            pca_mean,
            pca_basis,
            np.repeat(blocks[None], len(rows), axis=0),
            action_feature_chart,
        )
        prediction = _predict(model, x, int(pca_basis.shape[0]), device, payload["adapter_training"])
        decoded: dict[int, np.ndarray] = {}
        for horizon in HORIZONS:
            decoder, latent_center, target_center = readouts[horizon]
            decoded[horizon] = (
                (prediction[horizon] - latent_center.reshape(1, -1)) @ decoder
                + target_center.reshape(1, -1)
            )
        predicted_by_action.append(decoded)
    actual = _simulate_effects(rows, axis_actions)
    roots = [_root_key(row) for row in rows]
    summaries: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        predicted = np.stack([item[horizon] for item in predicted_by_action], axis=1)  # N,5,2
        actual_h = actual[:, :, horizon_index, :]
        pred_x = (predicted[:, 0] - predicted[:, 1]) / (2.0 * action_magnitude)
        pred_y = (predicted[:, 2] - predicted[:, 3]) / (2.0 * action_magnitude)
        actual_x = (actual_h[:, 0] - actual_h[:, 1]) / (2.0 * action_magnitude)
        actual_y = (actual_h[:, 2] - actual_h[:, 3]) / (2.0 * action_magnitude)
        pred_jacobian = np.stack([pred_x, pred_y], axis=-1)
        actual_jacobian = np.stack([actual_x, actual_y], axis=-1)
        pair_error = np.linalg.norm(pred_jacobian - actual_jacobian, axis=(1, 2))
        actual_scale = np.maximum(np.linalg.norm(actual_jacobian, axis=(1, 2)), 0.1)
        normalized_error = pair_error / actual_scale
        pred_flat = pred_jacobian.reshape(len(rows), -1)
        actual_flat = actual_jacobian.reshape(len(rows), -1)
        cosine = np.sum(pred_flat * actual_flat, axis=1) / np.maximum(
            np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(actual_flat, axis=1), 1e-8
        )
        zero_error = np.linalg.norm(predicted[:, 4] - actual_h[:, 4], axis=1)
        summaries[str(horizon)] = {
            "action_inverse_risk": _bootstrap_mean(normalized_error, roots, 1000 + int(horizon), bootstrap_samples),
            "effect_cosine": _bootstrap_mean(cosine, roots, 2000 + int(horizon), bootstrap_samples),
            "directional_identifiability_fraction": _bootstrap_mean((cosine > 0).astype(np.float64), roots, 3000 + int(horizon), bootstrap_samples),
            "zero_action_effect_error": _bootstrap_mean(zero_error, roots, 4000 + int(horizon), bootstrap_samples),
            "predicted_jacobian": _matrix_summary(pred_jacobian),
            "actual_jacobian": _matrix_summary(actual_jacobian),
            "support": {
                "root_count": len(set(roots)),
                "row_count": len(rows),
                "action_magnitude": float(action_magnitude),
                "action_fan": "+axis/-axis for each action coordinate plus zero",
            },
        }
    report = {
        "schema": "pointmaze_action_identifiability_audit.v1",
        "checkpoint": str(checkpoint.resolve()),
        "architecture": payload.get("architecture"),
        "seed": payload.get("seed"),
        "training_stage": payload.get("stage"),
        "environment": "PointMaze",
        "horizons": summaries,
        "protocol": {
            "fit_split": str(fit.get("split", {}).get("name", "unknown")),
            "outcome_split": str(outcome.get("split", {}).get("name", "unknown")),
            "external_root_disjoint": True,
            "external_rows_excluded_from_fit_and_outcome": True,
            "model_runtime_input": "RGB-derived source latent plus balanced action fan",
            "simulator_state_used_for": "offline labels only",
            "target_intercept": "train-only horizon-specific displacement center retained",
            "bootstrap_unit": "source root",
            "bootstrap_samples": int(bootstrap_samples),
        },
        "claim_boundary": "Independent action-identifiability diagnostic; not evidence of online planning or universal transfer.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "action_identifiability.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path, required=True)
    parser.add_argument("--outcome-export", type=Path, required=True)
    parser.add_argument("--external-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-magnitude", type=float, default=0.75)
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    report = audit(**vars(args))
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "horizons": {
            horizon: values["action_inverse_risk"] for horizon, values in report["horizons"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
