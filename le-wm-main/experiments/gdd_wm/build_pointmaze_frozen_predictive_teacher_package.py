"""Build a train-only, outcome-free predictive teacher package for PointMaze.

This is deliberately a narrow control.  It learns a frozen ridge transition
from a source latent and an action-sequence summary to the *already frozen
world-model terminal latent*.  Its score is the negative terminal-latent
prediction residual.  It never fits, normalizes, selects, or ranks from
``real_task_cost`` or another execution outcome.

The builder requires at least three independently generated candidate contexts
with common action anchors and different full action sets.  It writes the
metadata expected by the strict GDD D1/D2 routing audit, but it does not claim
that the predictive score is a useful teacher or authorize D2 by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_pointmaze_frozen_predictive_teacher.v0.1"
SPLITS = frozenset({"train", "calibration", "outcome"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _load_context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected a JSON object with metadata and records")
    if not payload["records"]:
        raise ValueError(f"{path}: records is empty")
    return dict(payload["metadata"]), [dict(row) for row in payload["records"]]


def _action_summary(row: dict[str, Any]) -> np.ndarray:
    sequence = np.asarray(row.get("action_sequence"), dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] < 1 or sequence.shape[1] < 1 or not np.isfinite(sequence).all():
        raise ValueError(f"{row.get('instance_id')}: action_sequence must be finite [steps, action_dim]")
    return np.concatenate((sequence.mean(axis=0), sequence.std(axis=0), sequence[-1], np.asarray([float(sequence.shape[0])])) )


def _source_latent(row: dict[str, Any]) -> np.ndarray:
    value = np.asarray(row.get("source_latent_model"), dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{row.get('instance_id')}: source_latent_model must be finite and non-empty")
    return value


def _terminal_latent(row: dict[str, Any], source_dataset: Path, cache: dict[Path, Any]) -> np.ndarray:
    reference = str(row.get("predicted_terminal_latent_ref", row.get("predicted_latent_ref", "")))
    if "arrays.npz::" not in reference:
        raise ValueError(f"{source_dataset}: {row.get('instance_id')} has no predicted terminal latent reference")
    arrays_path = source_dataset.parent / "arrays.npz"
    if arrays_path not in cache:
        if not arrays_path.exists():
            raise FileNotFoundError(f"{source_dataset}: missing {arrays_path.name}")
        cache[arrays_path] = np.load(arrays_path, allow_pickle=False)
    key = reference.split("arrays.npz::", 1)[1]
    arrays = cache[arrays_path]
    if key not in arrays.files:
        raise ValueError(f"{arrays_path}: missing terminal latent {key}")
    value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{arrays_path}: invalid terminal latent {key}")
    return value


def _fit_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate((np.ones((len(standardized), 1)), standardized), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets), mean, scale


def _predict(features: np.ndarray, beta: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / scale
    return np.concatenate((np.ones((len(standardized), 1)), standardized), axis=1) @ beta


def _validate_contexts(contexts: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]], minimum_contexts: int) -> dict[str, Any]:
    if len(contexts) < minimum_contexts:
        raise ValueError(f"at least {minimum_contexts} independent candidate contexts are required")
    environments = {str(metadata.get("environment", rows[0].get("environment", ""))) for _, metadata, rows in contexts}
    if environments != {"PointMaze"}:
        raise ValueError(f"contexts must all be PointMaze, got {sorted(environments)}")
    families = {str(metadata.get("candidate_family", rows[0].get("candidate_generator", ""))) for _, metadata, rows in contexts}
    if len(families) != 1 or not next(iter(families)):
        raise ValueError("candidate_family must be non-empty and identical across contexts")
    model_identities: set[str] = set()
    for _, metadata, _ in contexts:
        declared_hash = str(metadata.get("world_model_sha256", ""))
        checkpoint = str(metadata.get("checkpoint", ""))
        if declared_hash:
            model_identities.add(f"sha256:{declared_hash}")
        elif checkpoint:
            checkpoint_path = Path(checkpoint)
            model_identities.add(
                f"sha256:{_sha256_file(checkpoint_path)}" if checkpoint_path.exists() else f"declared:{checkpoint}"
            )
        else:
            raise ValueError("every context must declare a frozen world-model checkpoint or SHA256")
    if len(model_identities) != 1:
        raise ValueError("candidate contexts must use the same frozen world model")

    root_splits: list[dict[str, str]] = []
    action_sets: dict[str, list[set[str]]] = defaultdict(list)
    for path, _, rows in contexts:
        split_map: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        per_root_actions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            root, split = str(row.get("instance_id", "")), str(row.get("split", ""))
            if not root or split not in SPLITS:
                raise ValueError(f"{path}: every row requires a root and train/calibration/outcome split")
            previous = split_map.setdefault(root, split)
            if previous != split:
                raise ValueError(f"{path}: root {root} occurs in multiple splits")
            action_hash = _canonical_hash(row.get("action_sequence"))
            key = (root, action_hash)
            if key in seen:
                raise ValueError(f"{path}: duplicate root/action sequence {root}/{action_hash}")
            seen.add(key)
            per_root_actions[root].add(action_hash)
        root_splits.append(split_map)
        for root, actions in per_root_actions.items():
            action_sets[root].append(actions)
    reference = root_splits[0]
    for index, split_map in enumerate(root_splits[1:], start=1):
        if split_map != reference:
            raise ValueError(f"root split mismatch between context 0 and context {index}")
    if not {"train", "calibration", "outcome"}.issubset(set(reference.values())):
        raise ValueError("contexts require non-empty train, calibration, and outcome root splits")
    for root, sets in action_sets.items():
        if len(sets) != len(contexts):
            raise ValueError(f"root {root} is missing from a context")
        common = set.intersection(*sets)
        if len(common) < 2:
            raise ValueError(f"root {root} has fewer than two repeated action anchors")
        if len({tuple(sorted(values)) for values in sets}) < 2:
            raise ValueError(f"root {root} has no genuinely different candidate contexts")
    return {
        "root_splits": reference,
        "candidate_family": next(iter(families)),
        "action_sets": action_sets,
        "world_model_sha256": next(iter(model_identities)).split(":", 1)[1],
    }


def build(input_paths: list[Path], output: Path, *, ridge: float = 1.0e-3, minimum_contexts: int = 3) -> dict[str, Any]:
    """Build a package whose learned target is only frozen predicted latents."""
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("candidate context paths must be distinct")
    contexts = [(path.resolve(), *_load_context(path.resolve())) for path in input_paths]
    validated = _validate_contexts(contexts, minimum_contexts)
    cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context_manifest: list[dict[str, Any]] = []
    source_dimension: int | None = None
    action_dimension: int | None = None
    target_dimension: int | None = None
    for context_index, (path, metadata, source_rows) in enumerate(contexts):
        context_id = f"context_{context_index:02d}_{_sha256_file(path)[:12]}"
        context_manifest.append({
            "context_id": context_id,
            "candidate_dataset": str(path),
            "candidate_dataset_sha256": _sha256_file(path),
            "candidate_bank_hash": metadata.get("candidate_bank_hash", source_rows[0].get("candidate_bank_hash")),
            "record_count": len(source_rows),
        })
        for source_row in source_rows:
            source = _source_latent(source_row)
            action = _action_summary(source_row)
            terminal = _terminal_latent(source_row, path, cache)
            source_dimension = source.size if source_dimension is None else source_dimension
            action_dimension = action.size if action_dimension is None else action_dimension
            target_dimension = terminal.size if target_dimension is None else target_dimension
            if source.size != source_dimension:
                raise ValueError("source latent dimension differs across records; cannot fit a single frozen predictor")
            if action.size != action_dimension:
                raise ValueError("action summary dimension differs across records; cannot fit a single frozen predictor")
            if terminal.size != target_dimension:
                raise ValueError("predicted terminal latent dimension differs across records")
            row = dict(source_row)
            row.update({
                "candidate_id": str(source_row.get("candidate_id")),
                "candidate_bank_context_id": context_id,
                "teacher_set_id": f"{source_row['instance_id']}__{context_id}",
                "predicted_latent_source_dataset": str(path),
                "root_sha256": _canonical_hash(str(source_row["instance_id"])),
                "action_sequence_sha256": _canonical_hash(source_row["action_sequence"]),
            })
            rows.append(row)
            features.append(np.concatenate((source, action)))
            targets.append(terminal)
    feature_matrix = np.stack(features)
    target_matrix = np.stack(targets)
    train_indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    if len(train_indices) < feature_matrix.shape[1] + 1:
        raise ValueError("insufficient train rows for a train-only frozen predictive ridge")
    beta, mean, scale = _fit_ridge(feature_matrix[train_indices], target_matrix[train_indices], ridge)
    prediction = _predict(feature_matrix, beta, mean, scale)
    scores = -np.mean((prediction - target_matrix) ** 2, axis=1)
    for row, score in zip(rows, scores):
        row.update({
            "teacher_model_id": "pointmaze_train_only_frozen_predictive_latent_ridge_v1",
            "teacher_score_source": "train_only_predictive_latent_residual",
            "teacher_score": float(score),
            "teacher_fit_split": "train",
            "teacher_training_target_ref": "frozen_predicted_terminal_latent",
        })
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(row)
    for group in groups.values():
        for rank, row in enumerate(sorted(group, key=lambda item: (-float(item["teacher_score"]), str(item["candidate_id"])) )):
            row["teacher_rank"] = rank
    train_roots = sorted({str(row["instance_id"]) for row in rows if row["split"] == "train"})
    train_actions = {row["action_sequence_sha256"] for row in rows if row["split"] == "train"}
    held_actions = {row["action_sequence_sha256"] for row in rows if row["split"] == "outcome"}
    novel_fraction = float(len(held_actions - train_actions) / len(held_actions)) if held_actions else 0.0
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "teacher_model.npz", beta=beta, mean=mean, scale=scale)
    teacher_checkpoint_sha256 = _sha256_file(output / "teacher_model.npz")
    routing_signature = {
        "environment": "PointMaze",
        "world_model_sha256": validated["world_model_sha256"],
        "horizon": int(next(iter({int(row.get("predicted_horizon", 0)) for row in rows}))),
        "candidate_family": validated["candidate_family"],
        "action_frame": "world_fixed",
        "task_cost_version": "external_audit_only_v1",
    }
    metadata = {
        "schema": SCHEMA,
        "teacher_model_id": "pointmaze_train_only_frozen_predictive_latent_ridge_v1",
        "teacher_model_frozen_before_d1": True,
        "teacher_fit_split": "train",
        "teacher_fit_root_ids": train_roots,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_input_fields": ["source_latent_model", "action_sequence_summary"],
        "teacher_training_target_fields": ["predicted_terminal_latent"],
        "teacher_training_uses_execution_outcomes": False,
        "teacher_score_semantics": "negative frozen terminal-latent prediction residual",
        "candidate_dataset_input_sha256": _canonical_hash([entry["candidate_dataset_sha256"] for entry in context_manifest]),
        "context_manifest": context_manifest,
        "candidate_context_count_required": minimum_contexts,
        "held_candidate_min_novel_fraction": 0.25,
        "held_candidate_novel_fraction": novel_fraction,
        "routing_signature": routing_signature,
        "fit_provenance": {
            "fit_row_count": int(len(train_indices)),
            "feature_dimension": int(feature_matrix.shape[1]),
            "target_dimension": int(target_matrix.shape[1]),
            "ridge": float(ridge),
            "outcome_fields_read_for_fit": [],
            "candidate_id_read_for_fit": False,
            "fit_selection_uses_calibration_or_outcome": False,
        },
        "claim_boundary": (
            "Outcome-free frozen predictive confidence control only. It is neither an independent "
            "D-JEPA preference teacher nor evidence of D2 authorization, geometric repair, or planning benefit."
        ),
    }
    package_path = output / "candidate_dataset.json"
    package_path.write_text(json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score_hash = _canonical_hash([float(value) for value in scores])
    report = {
        "schema": SCHEMA,
        "status": "built_valid_predictive_teacher_package",
        "package": str(package_path),
        "summary": {
            "record_count": len(rows),
            "train_root_count": len(train_roots),
            "context_count": len(context_manifest),
            "held_candidate_novel_fraction": novel_fraction,
            "teacher_score_sha256": score_hash,
            "train_terminal_latent_rmse": float(np.sqrt(np.mean((prediction[train_indices] - target_matrix[train_indices]) ** 2))),
        },
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="independent PointMaze candidate_dataset.json; repeat at least three times")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--minimum-contexts", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, ridge=args.ridge, minimum_contexts=args.minimum_contexts), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
