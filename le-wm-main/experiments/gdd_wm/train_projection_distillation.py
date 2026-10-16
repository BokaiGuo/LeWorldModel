"""Run fail-closed D2 projection-only geometry distillation.

The runner deliberately refuses to train when the D1 teacher audit is not
passed.  A valid feature table is an NPZ with string ``keys`` matching
``instance_id::teacher_set_id::candidate_id`` and float ``features`` and
``goal_features`` arrays of shape ``[records, dimension]``.  Prediction and
independent planning audits are separate JSON gates; they are never inferred
from the teacher or from the same outcome labels used for fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _record_key(row: dict[str, Any]) -> str:
    return "::".join((str(row["instance_id"]), str(row["teacher_set_id"]), str(row["candidate_id"])))


FEATURE_TABLE_SCHEMA = "gdd_wm_student_feature_table.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature_table(path: Path, *, expected_dataset_sha256: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"keys", "features", "goal_features", "metadata_json"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"feature table is missing arrays, including provenance: {missing}")
        keys = np.asarray(data["keys"])
        features = np.asarray(data["features"], dtype=np.float64)
        goals = np.asarray(data["goal_features"], dtype=np.float64)
        metadata_values = np.asarray(data["metadata_json"])
    if metadata_values.size != 1:
        raise ValueError("feature table metadata_json must contain exactly one JSON object")
    metadata_value = metadata_values.reshape(-1)[0]
    if isinstance(metadata_value, bytes):
        metadata_value = metadata_value.decode("utf-8")
    try:
        metadata = json.loads(str(metadata_value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("feature table metadata_json must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("feature table metadata_json must encode a JSON object")
    required_metadata = {
        "schema": FEATURE_TABLE_SCHEMA,
        "dataset_sha256": expected_dataset_sha256,
        "key_schema": "instance_id::teacher_set_id::candidate_id",
        "candidate_set_context_input_forbidden": True,
        "teacher_rank_input_forbidden": True,
        "held_out_outcome_input_forbidden": True,
        "dynamics_frozen": True,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"feature table provenance.{key} must be {expected!r}")
    keys = np.asarray([item.decode() if isinstance(item, bytes) else str(item) for item in keys])
    if keys.ndim != 1 or len(np.unique(keys)) != len(keys):
        raise ValueError("feature table keys must be one-dimensional and unique")
    if features.ndim != 2 or goals.shape != features.shape or len(keys) != len(features):
        raise ValueError("features and goal_features must have matching [records, dimension] shapes")
    if not np.isfinite(features).all() or not np.isfinite(goals).all():
        raise ValueError("feature table contains non-finite values")
    return {"keys": keys, "features": features, "goal_features": goals, "metadata": metadata}


def _gate(path: Path | None, name: str) -> dict[str, Any]:
    if path is None:
        return {"name": name, "status": "missing", "passed": False, "source": None}
    payload = _load_json(path)
    passed = bool(payload.get("passed", False) or payload.get("status") in {"passed", "certified"})
    return {"name": name, "status": str(payload.get("status", "missing")), "passed": passed, "source": str(path)}


def _fit_projection(
    features: np.ndarray,
    goals: np.ndarray,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    rank: int,
    epochs: int,
    learning_rate: float,
    margin: float,
    temperature: float,
    ridge: float,
) -> np.ndarray:
    dimension = features.shape[1]
    max_rank = min(dimension, int(len(indices)))
    if not 1 <= rank <= max_rank:
        raise ValueError(
            "projection rank must lie in [1, min(feature dimension, train sample count)]"
        )
    centered = features[indices] - goals[indices]
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = vectors[:rank].copy()
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices.tolist():
        row = rows[index]
        groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(index)
    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1:]:
                delta = float(rows[left]["teacher_score"]) - float(rows[right]["teacher_score"])
                if abs(delta) <= margin:
                    continue
                pairs.append((left, right) if delta > 0.0 else (right, left))
    if not pairs:
        raise ValueError("D2 training split contains no comparable teacher pairs")
    for _ in range(int(epochs)):
        gradient = ridge * projection
        for preferred, rejected in pairs:
            preferred_delta = projection @ (features[preferred] - goals[preferred])
            rejected_delta = projection @ (features[rejected] - goals[rejected])
            preferred_cost = float(preferred_delta @ preferred_delta) / rank
            rejected_cost = float(rejected_delta @ rejected_delta) / rank
            violation = (margin + preferred_cost - rejected_cost) / temperature
            violation = float(np.clip(violation, -60.0, 60.0))
            weight = 1.0 / (1.0 + np.exp(-violation)) / temperature
            gradient += weight * (
                2.0 * np.outer(preferred_delta, features[preferred] - goals[preferred]) / rank
                - 2.0 * np.outer(rejected_delta, features[rejected] - goals[rejected]) / rank
            )
        projection -= float(learning_rate) * gradient / max(len(pairs), 1)
        # Keep the projection scale identifiable and prevent a trivial scale
        # collapse from satisfying the pairwise margin.
        singular = np.linalg.svd(projection, compute_uv=False)
        projection /= max(float(np.mean(singular)), 1.0e-8)
    return projection


def _evaluate(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    goals: np.ndarray,
    indices: np.ndarray,
    projection: np.ndarray,
    margin: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in indices.tolist():
        row = rows[index]
        groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(index)
    pair_correct = pair_total = 0
    regrets: list[float] = []
    top1_hits = 0
    group_count = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group_count += 1
        costs = np.asarray([
            np.linalg.norm(projection @ (features[index] - goals[index])) ** 2 / projection.shape[0]
            for index in group
        ])
        teacher_scores = np.asarray([float(rows[index]["teacher_score"]) for index in group])
        for left_pos, left in enumerate(group):
            for right_pos in range(left_pos + 1, len(group)):
                right = group[right_pos]
                teacher_delta = teacher_scores[left_pos] - teacher_scores[right_pos]
                if abs(teacher_delta) <= margin:
                    continue
                pair_total += 1
                predicted_delta = costs[left_pos] - costs[right_pos]
                pair_correct += int((teacher_delta > 0.0 and predicted_delta < 0.0) or (teacher_delta < 0.0 and predicted_delta > 0.0))
        teacher_best = int(np.argmax(teacher_scores))
        predicted_best = int(np.argmin(costs))
        top1_hits += int(teacher_best == predicted_best)
        real_costs = np.asarray([float(rows[index]["real_task_cost"]) for index in group])
        regrets.append(float(real_costs[predicted_best] - np.min(real_costs)))
    return {
        "n": int(len(indices)),
        "group_count": int(group_count),
        "pairwise_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
        "comparable_pairs": int(pair_total),
        "top1_teacher_agreement": None if group_count == 0 else float(top1_hits / group_count),
        "mean_real_task_regret": None if not regrets else float(np.mean(regrets)),
        "regret_values": regrets,
    }


def run(
    *, dataset_path: Path, d1_path: Path, output_path: Path,
    feature_table_path: Path | None = None,
    prediction_audit_path: Path | None = None,
    planning_audit_path: Path | None = None,
    projection_rank: int = 2, epochs: int = 200, learning_rate: float = 0.05,
    margin: float = 1.0e-3, temperature: float = 0.1, ridge: float = 1.0e-4,
) -> dict[str, Any]:
    d1 = _load_json(d1_path)
    report: dict[str, Any] = {
        "schema": "gdd_wm_d2_projection_distillation.v0.1",
        "status": "blocked_d1_gate",
        "protocol": {
            "dataset": str(dataset_path),
            "d1_report": str(d1_path),
            "feature_table": None if feature_table_path is None else str(feature_table_path),
            "projection_type": "linear_train_only",
            "dynamics_frozen": True,
            "teacher_runtime_forbidden": True,
            "candidate_set_context_input_forbidden": True,
            "outcome_used_for_projection_selection": False,
            "projection_rank": int(projection_rank),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "pair_margin": float(margin),
        },
        "gates": {
            "d1": {"status": d1.get("status"), "passed": d1.get("status") == "passed"},
            "prediction_preservation": _gate(prediction_audit_path, "prediction_preservation"),
            "independent_planning_consequence": _gate(planning_audit_path, "independent_planning_consequence"),
        },
        "claim_boundary": "Projection-only D2 gate; no D-JEPA or planning claim is released without every independent gate.",
    }
    if d1.get("status") != "passed":
        report["reason"] = "D1 teacher stability/execution gate is not passed; projection training was not executed"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "d2_projection_distillation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    if feature_table_path is None:
        report["status"] = "blocked_missing_student_features"
        report["reason"] = "valid D1 exists but no predicted/goal feature table was supplied"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "d2_projection_distillation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    dataset = _load_json(dataset_path)
    rows = dataset.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate dataset must contain a non-empty records list")
    dataset_sha256 = _sha256(dataset_path)
    feature_table_sha256 = _sha256(feature_table_path)
    report["protocol"]["dataset_sha256"] = dataset_sha256
    report["protocol"]["feature_table_sha256"] = feature_table_sha256
    table = _load_feature_table(feature_table_path, expected_dataset_sha256=dataset_sha256)
    lookup = {key: index for index, key in enumerate(table["keys"])}
    row_indices = []
    for row in rows:
        key = _record_key(row)
        if key not in lookup:
            raise ValueError(f"feature table is missing record key {key}")
        row_indices.append(lookup[key])
    features = table["features"][row_indices]
    goals = table["goal_features"][row_indices]
    train = np.asarray([index for index, row in enumerate(rows) if row.get("split") == "train"], dtype=np.int64)
    evaluation_splits = sorted({str(row.get("split")) for row in rows if row.get("split") != "train"})
    projection = _fit_projection(features, goals, rows, train, projection_rank, epochs, learning_rate, margin, temperature, ridge)
    baseline = np.eye(features.shape[1], dtype=np.float64)
    report["status"] = "completed_diagnostic"
    report["projection"] = projection.tolist()
    report["feature_table_provenance"] = table["metadata"]
    report["metrics"] = {
        "train": {"fitted_projection": _evaluate(rows, features, goals, train, projection, margin), "frozen_baseline": _evaluate(rows, features, goals, train, baseline, margin)},
        "held_out": {
            split: {
                "fitted_projection": _evaluate(rows, features, goals, np.asarray([i for i, row in enumerate(rows) if row.get("split") == split], dtype=np.int64), projection, margin),
                "frozen_baseline": _evaluate(rows, features, goals, np.asarray([i for i, row in enumerate(rows) if row.get("split") == split], dtype=np.int64), baseline, margin),
            }
            for split in evaluation_splits
        },
    }
    report["permission"] = {
        "prediction_preservation": report["gates"]["prediction_preservation"]["passed"],
        "independent_planning_consequence": report["gates"]["independent_planning_consequence"]["passed"],
        "status": "permitted" if all(item["passed"] for item in report["gates"].values()) else "rejected_missing_independent_gate",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "d2_projection_distillation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--d1-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--prediction-audit", type=Path, default=None)
    parser.add_argument("--planning-audit", type=Path, default=None)
    parser.add_argument("--projection-rank", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    args = parser.parse_args()
    report = run(
        dataset_path=args.dataset, d1_path=args.d1_report, output_path=args.output,
        feature_table_path=args.feature_table, prediction_audit_path=args.prediction_audit,
        planning_audit_path=args.planning_audit, projection_rank=args.projection_rank,
        epochs=args.epochs, learning_rate=args.learning_rate, margin=args.margin,
        temperature=args.temperature, ridge=args.ridge,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"completed_diagnostic", "blocked_d1_gate", "blocked_missing_student_features"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
