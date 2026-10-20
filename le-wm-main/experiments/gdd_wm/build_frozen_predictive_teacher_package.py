"""Build an execution-independent frozen predictive teacher package.

The existing GDD-WM candidate banks used ``teacher_score = -real_task_cost``.
That is useful as an oracle control, but it cannot validate a teacher.  This
builder fits a small ridge readout on *training-root* predicted terminal
latents and their training execution costs, freezes the readout, and scores
all candidate rows without reading outcome costs.  Outcome costs are copied
only as audit targets and are never used for fitting or ranking.

This is a legitimate frozen predictive-teacher control, not a D-JEPA claim.
It becomes an execution-validated teacher only after an independent successor
reference join and held-out execution audit pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


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


def _latent(row: dict[str, Any], dataset_path: Path, cache: dict[Path, Any]) -> np.ndarray:
    ref = str(row.get("predicted_terminal_latent_ref", row.get("predicted_latent_ref", "")))
    if "arrays.npz::" not in ref:
        raise ValueError(f"missing predicted latent reference for {row.get('instance_id')}")
    key = ref.split("arrays.npz::", 1)[1]
    arrays_path = dataset_path.parent / "arrays.npz"
    if arrays_path not in cache:
        cache[arrays_path] = np.load(arrays_path, allow_pickle=False)
    arrays = cache[arrays_path]
    if key not in arrays.files:
        raise ValueError(f"{arrays_path}: missing array {key}")
    value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"invalid predicted latent {key}")
    goal = np.asarray(row.get("goal_raw", []), dtype=np.float64).reshape(-1)
    if goal.size == 0 or not np.isfinite(goal).all():
        raise ValueError(f"missing goal_raw for {row.get('instance_id')}")
    return np.concatenate([value, goal])


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    xz = (x - mean) / scale
    design = np.concatenate([np.ones((len(xz), 1)), xz], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return beta, mean, scale


def build(input_path: Path, output_dir: Path, *, ridge: float = 1.0e-3) -> dict[str, Any]:
    metadata, rows = _load(input_path)
    if not rows:
        raise ValueError("empty candidate package")
    declared_history_windows = int(metadata.get("history_windows", 1))
    predictor_history_windows = int(metadata.get("predictor_history_windows", 1))
    if declared_history_windows < 1 or predictor_history_windows < 1:
        raise ValueError("history window metadata must be positive")
    cache: dict[Path, Any] = {}
    features: list[np.ndarray] = []
    for row in rows:
        source = Path(str(row.get("predicted_latent_source_dataset", row.get("source_candidate_dataset", input_path))))
        if not source.is_absolute():
            source = Path.cwd() / source
        features.append(_latent(row, source, cache))
    x = np.stack(features)
    train_indices = np.asarray(
        [i for i, row in enumerate(rows) if str(row.get("split")) == "train"], dtype=np.int64
    )
    if len(train_indices) < 4:
        raise ValueError("at least four train rows are required")
    y_train = np.asarray([float(rows[i]["real_task_cost"]) for i in train_indices], dtype=np.float64)
    if not np.isfinite(y_train).all():
        raise ValueError("non-finite train costs")
    beta, mean, scale = _ridge_fit(x[train_indices], y_train, ridge)
    design = np.concatenate([np.ones((len(x), 1)), (x - mean) / scale], axis=1)
    predicted_cost = design @ beta

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["teacher_set_id"]), []).append(index)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["teacher_model_id"] = "frozen_predicted_latent_ridge_v1"
        updated["teacher_score_source"] = "train_root_predicted_terminal_latent_ridge"
        updated["teacher_score"] = float(-predicted_cost[index])
        updated["teacher_fit_split"] = "train"
        updated["teacher_outcome_cost_used_for_fit"] = bool(str(row.get("split")) == "train")
        out_rows.append(updated)
    for set_id, indices in grouped.items():
        order = sorted(indices, key=lambda i: (-float(out_rows[i]["teacher_score"]), str(out_rows[i]["candidate_id"])))
        for rank, index in enumerate(order, start=1):
            out_rows[index]["teacher_rank"] = rank

    out_metadata = dict(metadata)
    out_metadata.update({
        "teacher_model_id": "frozen_predicted_latent_ridge_v1",
        "teacher_model_frozen_before_distillation": True,
        "teacher_score_source": "train_root_predicted_terminal_latent_ridge",
        "teacher_fit_input_forbidden": ["outcome", "environment_test"],
        "teacher_runtime_input_forbidden": True,
        "teacher_outcome_cost_used_for_fit": "train_only",
        "source_dataset_sha256": _sha256(input_path),
        "ridge": float(ridge),
        "teacher_declared_history_windows": declared_history_windows,
        "teacher_predictor_history_windows": predictor_history_windows,
        "teacher_history_compatible": declared_history_windows == predictor_history_windows,
        "teacher_history_contract": (
            "predicted_terminal_latent_uses_declared_history"
            if declared_history_windows == predictor_history_windows
            else "predicted_terminal_latent_uses_predictor_history_not_candidate_history"
        ),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    source_arrays = input_path.parent / "arrays.npz"
    packaged_arrays = output_dir / "arrays.npz"
    if not source_arrays.exists():
        raise FileNotFoundError(f"candidate package arrays are missing: {source_arrays}")
    shutil.copyfile(source_arrays, packaged_arrays)
    out_metadata["packaged_arrays_sha256"] = _sha256(packaged_arrays)
    out_metadata["packaged_arrays_source"] = str(source_arrays.resolve())
    payload = {"metadata": out_metadata, "records": out_rows}
    output_path = output_dir / "candidate_dataset.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "teacher_model.npz", beta=beta, mean=mean, scale=scale)
    report = {
        "schema": "gdd_wm_frozen_predictive_teacher_package.v0.1",
        "status": "built",
        "summary": {
            "record_count": len(out_rows),
            "train_fit_rows": int(len(train_indices)),
            "context_count": len({str(row.get("candidate_bank_context_id")) for row in out_rows}),
            "predicted_cost_train_rmse": float(np.sqrt(np.mean((predicted_cost[train_indices] - y_train) ** 2))),
            "output_candidate_dataset": str(output_path),
            "declared_history_windows": declared_history_windows,
            "predictor_history_windows": predictor_history_windows,
            "history_compatible": declared_history_windows == predictor_history_windows,
            "packaged_arrays": str(packaged_arrays),
        },
        "claim_boundary": "frozen predictive teacher control; not D-JEPA validation or closed-loop planning",
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, ridge=args.ridge), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
