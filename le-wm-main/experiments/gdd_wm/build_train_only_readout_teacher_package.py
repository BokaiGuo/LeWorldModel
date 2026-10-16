"""Build a non-D-JEPA train-only readout teacher control package.

This is an infrastructure positive control for the D1 protocol.  It joins
identical root/action records from independent candidate-bank contexts, fits a
frozen standardized ridge readout using *train-root* predicted latents only,
and writes fresh teacher scores without copying the source cost oracle.  The
result is intentionally not a D-JEPA teacher and is never an authorization for
D2 distillation; it only tests whether repeated-context and runtime-reference
gates can operate on a non-leaking control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_train_only_readout_teacher_control.v0.1"
FEATURE_FAMILIES = (
    "predicted_terminal",
    "source_plus_terminal",
    "source_plus_delta",
    "source_delta_action",
    "action_summary",
)


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected JSON object with records")
    return dict(payload.get("metadata", {})), list(payload["records"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_hash(row: dict[str, Any]) -> str:
    encoded = json.dumps(row["action_sequence"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _feature_key(row: dict[str, Any]) -> str:
    ref = str(row.get("predicted_latent_ref", ""))
    if "::" not in ref:
        raise ValueError(f"predicted_latent_ref has no array key: {ref!r}")
    return ref.split("::", 1)[1]


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("ridge inputs have incompatible shapes")
    if len(features) == 0:
        raise ValueError("train split is empty")
    mean = np.mean(features, axis=0)
    scale = np.maximum(np.std(features, axis=0), 1.0e-6)
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + float(alpha) * penalty,
        design.T @ target,
    )
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _predict(features: np.ndarray, fit: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - fit["mean"]) / fit["scale"]
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    return design @ fit["coefficients"]


def _action_summary(action_sequence: Any) -> np.ndarray:
    actions = np.asarray(action_sequence, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError("action_sequence must be a non-empty rank-2 array")
    midpoint = max(1, actions.shape[0] // 2)
    second_half = actions[midpoint:] if midpoint < actions.shape[0] else actions[-1:]
    return np.concatenate([
        np.mean(actions, axis=0),
        np.std(actions, axis=0),
        actions[0],
        actions[-1],
        np.mean(actions[:midpoint], axis=0),
        np.mean(second_half, axis=0),
    ])


def _record_features(row: dict[str, Any], predicted: np.ndarray, family: str) -> np.ndarray:
    if family == "predicted_terminal":
        return predicted
    if family == "action_summary":
        return _action_summary(row["action_sequence"])
    source = np.asarray(row.get("source_latent_model"), dtype=np.float64)
    if source.ndim != 1 or source.shape != predicted.shape:
        raise ValueError("source_latent_model and predicted latent dimensions do not match")
    delta = predicted - source
    if family == "source_plus_terminal":
        return np.concatenate([source, predicted])
    if family == "source_plus_delta":
        return np.concatenate([source, delta])
    if family == "source_delta_action":
        return np.concatenate([source, delta, _action_summary(row["action_sequence"])])
    raise ValueError(f"unknown feature family: {family}")


def build(
    *, candidate_datasets: list[Path], arrays_paths: list[Path], output: Path,
    alpha: float = 1.0e-2, feature_family: str = "predicted_terminal",
    residualize_action: bool = False,
) -> dict[str, Any]:
    if len(candidate_datasets) < 2 or len(candidate_datasets) != len(arrays_paths):
        raise ValueError("at least two candidate datasets and one arrays file per dataset are required")
    if feature_family not in FEATURE_FAMILIES:
        raise ValueError(f"unknown feature family {feature_family!r}; choose from {FEATURE_FAMILIES}")
    loaded = [_load(path) for path in candidate_datasets]
    source_maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for path, (_, rows) in zip(candidate_datasets, loaded):
        current: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if "instance_id" not in row or "action_sequence" not in row:
                raise ValueError(f"{path}: instance_id/action_sequence missing")
            key = (str(row["instance_id"]), _action_hash(row))
            if key in current:
                raise ValueError(f"{path}: duplicate root/action pair {key}")
            current[key] = row
        source_maps.append(current)
    common = set(source_maps[0])
    for current in source_maps[1:]:
        common &= set(current)
    if not common:
        raise ValueError("candidate datasets have no common root/action records")

    arrays = [np.load(path, allow_pickle=False) for path in arrays_paths]
    contexts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (path, (metadata, _)) in enumerate(zip(candidate_datasets, loaded)):
        source_hash = str(metadata.get("candidate_bank_hash", ""))
        if not source_hash:
            source_hash = _sha256(path)[:20]
        context_id = f"readout_context_{index:02d}_{source_hash}"
        contexts.append({
            "context_id": context_id,
            "source_candidate_dataset": str(path.resolve()),
            "source_candidate_dataset_sha256": _sha256(path),
            "source_teacher_model_id": sorted({str(row.get("teacher_model_id", "missing")) for row in loaded[index][1]}),
        })
        for key in sorted(common):
            source = dict(source_maps[index][key])
            root, action_hash = key
            source["instance_id"] = root
            source["candidate_id"] = f"action_{action_hash}"
            source["matched_action_hash"] = action_hash
            source["candidate_bank_context_id"] = context_id
            source["teacher_set_id"] = f"{root}__{context_id}"
            source["teacher_model_id"] = "train_only_predicted_latent_ridge_v1"
            source["teacher_score_source"] = "train_root_predicted_latent_readout"
            source["predicted_latent_ref"] = f"{context_id}::{source['predicted_latent_ref']}"
            if source.get("realized_latent_ref") is not None:
                source["realized_latent_ref"] = f"{context_id}::{source['realized_latent_ref']}"
            rows.append(source)

    train_rows = [row for row in rows if str(row.get("split")) == "train"]
    heldout_rows = [row for row in rows if str(row.get("split")) != "train"]
    if not train_rows or not heldout_rows:
        raise ValueError("both train and held-out rows are required")
    features: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    targets: list[float] = []
    for row in train_rows:
        context_id = str(row["candidate_bank_context_id"])
        context_index = next(index for index, item in enumerate(contexts) if item["context_id"] == context_id)
        key = _feature_key(row)
        key = key.split("::", 1)[-1] if "::" in key else key
        if key not in arrays[context_index].files:
            raise ValueError(f"missing predicted latent array {key} in context {context_id}")
        value = np.asarray(arrays[context_index][key], dtype=np.float64)
        if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid predicted latent array {key}")
        features.append(_record_features(row, value, feature_family))
        action_features.append(_record_features(row, value, "action_summary"))
        targets.append(float(row["real_task_cost"]))
    train_features = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    action_fit = None
    if residualize_action:
        action_fit = _fit_ridge(np.asarray(action_features, dtype=np.float64), target_array, alpha)
        fit_target = target_array - _predict(np.asarray(action_features, dtype=np.float64), action_fit)
    else:
        fit_target = target_array
    fit = _fit_ridge(train_features, fit_target, alpha)

    all_features: list[np.ndarray] = []
    all_action_features: list[np.ndarray] = []
    for row in rows:
        context_id = str(row["candidate_bank_context_id"])
        context_index = next(index for index, item in enumerate(contexts) if item["context_id"] == context_id)
        key = _feature_key(row)
        key = key.split("::", 1)[-1] if "::" in key else key
        value = np.asarray(arrays[context_index][key], dtype=np.float64)
        all_features.append(_record_features(row, value, feature_family))
        all_action_features.append(_record_features(row, value, "action_summary"))
    predicted_cost = _predict(np.asarray(all_features, dtype=np.float64), fit)
    if action_fit is not None:
        predicted_cost += _predict(np.asarray(all_action_features, dtype=np.float64), action_fit)
    for row, cost in zip(rows, predicted_cost):
        row["teacher_score"] = float(-cost)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(row)
    for group in grouped.values():
        order = sorted(group, key=lambda item: (-float(item["teacher_score"]), str(item["candidate_id"])))
        for rank, row in enumerate(order):
            row["teacher_rank"] = int(rank)

    metadata = {
        "schema": SCHEMA,
        "teacher_model_id": "train_only_predicted_latent_ridge_v1",
        "feature_family": feature_family,
        "target_mode": "residualized_action_plus_feature" if residualize_action else "direct_task_cost",
        "teacher_fit_split": "train",
        "teacher_fit_root_count": len({str(row["instance_id"]) for row in train_rows}),
        "candidate_bank_context_count": len(contexts),
        "matched_root_action_pair_count": len(common),
        "outcome_used_for_teacher_or_metric_selection": False,
        "snapshot_selection_from_outcome": False,
        "teacher_frozen_before_distillation": True,
        "shared_candidate_bank": True,
        "runtime_teacher_forbidden": True,
        "contexts": contexts,
        "ridge_alpha": float(alpha),
        "claim_boundary": (
            "non-D-JEPA train-only readout control; D1 infrastructure positive control only; "
            "not teacher correctness, D2 authorization, or planning evidence"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": rows}
    output_path = output / "candidate_dataset.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_predicted_cost = _predict(train_features, fit)
    if action_fit is not None:
        train_predicted_cost += _predict(np.asarray(action_features, dtype=np.float64), action_fit)
    report = {
        "schema": SCHEMA,
        "status": "completed_control_package",
        "output": str(output_path),
        "metadata": metadata,
        "summary": {
            "record_count": len(rows),
            "train_record_count": len(train_rows),
            "heldout_record_count": len(heldout_rows),
            "root_count": len({str(row["instance_id"]) for row in rows}),
            "context_count": len(contexts),
            "matched_root_action_pair_count": len(common),
            "train_cost_rmse": float(np.sqrt(np.mean((
                train_predicted_cost - target_array
            ) ** 2))),
            "feature_dimension": int(train_features.shape[1]),
            "action_baseline_train_rmse": None if action_fit is None else float(np.sqrt(np.mean((
                _predict(np.asarray(action_features, dtype=np.float64), action_fit) - target_array
            ) ** 2))),
        },
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", action="append", required=True)
    parser.add_argument("--arrays", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--feature-family", choices=FEATURE_FAMILIES, default="predicted_terminal")
    parser.add_argument("--residualize-action", action="store_true")
    args = parser.parse_args()
    report = build(
        candidate_datasets=[Path(path) for path in args.candidate_dataset],
        arrays_paths=[Path(path) for path in args.arrays],
        output=args.output,
        alpha=args.alpha,
        feature_family=args.feature_family,
        residualize_action=args.residualize_action,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
