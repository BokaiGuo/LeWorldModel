"""Group-held-out prediction audit for full action-opportunity descriptors.

The existing Stage-1 panels already contain separate A and B probe vectors.
This audit asks whether the complete A vector predicts held-out B quality
better than the scalar A maximum. It is diagnostic only: B is a finite
held-out probe bank, not an optimal suffix oracle or physical outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_action_opportunity_stats import spearman


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "fixed_probe": ROOT / "results/pc_action_opportunity_descriptor_dev_20261104_codex/result.json",
    "random_probe": ROOT / "results/pc_action_opportunity_random_probe_dev_20261104_codex/result.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(payload: dict) -> list[dict]:
    return [row for row in payload.get("rows", []) if row.get("status") == "completed"]


def _features(rows: list[dict], mode: str, permutation: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray([row["a_improvements"] for row in rows], dtype=np.float64)
    if mode == "scalar":
        core = vectors.max(axis=1, keepdims=True)
    elif mode == "vector":
        core = vectors
    elif mode == "permuted_vector":
        if permutation is None:
            raise ValueError("permutation required")
        core = vectors[:, permutation]
    else:
        raise ValueError(mode)
    controls = [np.asarray([float(row["immediate_score"]) for row in rows])]
    if all("baseline_score" in row for row in rows):
        controls.append(np.asarray([float(row["baseline_score"]) for row in rows]))
    controls.append(np.asarray([15.0 - 5.0 * int(row["depth"]) for row in rows]))
    return np.column_stack([core, *controls])


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_scaled, test_scaled = _standardize(train_x, test_x)
    train_design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    test_design = np.column_stack([np.ones(len(test_scaled)), test_scaled])
    penalty = np.eye(train_design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + 1.0 * penalty,
        train_design.T @ train_y,
    )
    return test_design @ coefficients


def _evaluate(rows: list[dict], permutation: np.ndarray) -> dict:
    roots = sorted({int(row["root"]) for row in rows})
    target = np.asarray([max(row["b_improvements"]) for row in rows], dtype=np.float64)
    outputs: dict[str, list[float]] = {"scalar": [], "vector": [], "permuted_vector": []}
    truths: list[float] = []
    root_records = []
    for root in roots:
        test_mask = np.asarray([int(row["root"]) == root for row in rows])
        train_mask = ~test_mask
        if train_mask.sum() < 3 or not test_mask.any():
            continue
        predictions = {}
        for mode in outputs:
            predictions[mode] = _ridge_predict(
                _features([row for row, keep in zip(rows, train_mask) if keep], mode, permutation),
                target[train_mask],
                _features([row for row, keep in zip(rows, test_mask) if keep], mode, permutation),
            )
            outputs[mode].extend(predictions[mode].tolist())
        test_target = target[test_mask]
        truths.extend(test_target.tolist())
        root_records.append(
            {
                "root": root,
                "count": int(test_mask.sum()),
                "spearman_scalar": spearman(predictions["scalar"], test_target),
                "spearman_vector": spearman(predictions["vector"], test_target),
                "spearman_permuted_vector": spearman(
                    predictions["permuted_vector"], test_target
                ),
                "mae_scalar": float(np.mean(np.abs(predictions["scalar"] - test_target))),
                "mae_vector": float(np.mean(np.abs(predictions["vector"] - test_target))),
                "mae_permuted_vector": float(
                    np.mean(np.abs(predictions["permuted_vector"] - test_target))
                ),
            }
        )
    truth = np.asarray(truths, dtype=np.float64)
    return {
        "row_count": len(rows),
        "root_count": len(root_records),
        "candidate_level": {
            "spearman_scalar": spearman(np.asarray(outputs["scalar"]), truth),
            "spearman_vector": spearman(np.asarray(outputs["vector"]), truth),
            "spearman_permuted_vector": spearman(
                np.asarray(outputs["permuted_vector"]), truth
            ),
            "mae_scalar": float(np.mean(np.abs(np.asarray(outputs["scalar"]) - truth))),
            "mae_vector": float(np.mean(np.abs(np.asarray(outputs["vector"]) - truth))),
            "mae_permuted_vector": float(
                np.mean(np.abs(np.asarray(outputs["permuted_vector"]) - truth))
            ),
        },
        "root_records": root_records,
        "interpretation_boundary": (
            "held-out B probe quality only; not optimal suffix, physical outcome, "
            "independent confirmation, or planner evidence"
        ),
    }


def audit(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text())
    rows = _rows(payload)
    if not rows:
        raise ValueError(f"no completed rows in {input_path}")
    width = len(rows[0]["a_improvements"])
    permutation = np.arange(width - 1, -1, -1, dtype=np.int64)
    result = {
        "schema": "pc_action_opportunity_vector_prediction_audit.v1",
        "status": "completed",
        "source": str(input_path),
        "source_sha256": _sha256(input_path),
        "source_schema": payload.get("schema"),
        "not_rerun": True,
        "not_independent_confirmation": True,
        "feature_contract": {
            "target": "max(b_improvements)",
            "scalar": "max(a_improvements) plus immediate/baseline/horizon controls",
            "vector": "full a_improvements plus the same controls",
            "permuted_vector": "reverse action/probe column identity plus the same controls",
            "group_split": "leave_one_root_out",
        },
        "result": _evaluate(rows, permutation),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-input", type=Path, default=INPUTS["fixed_probe"])
    parser.add_argument("--random-input", type=Path, default=INPUTS["random_probe"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "pc_action_opportunity_vector_prediction_panel.v1",
        "status": "completed",
    }
    inputs = {"fixed_probe": args.fixed_input, "random_probe": args.random_input}
    for name, input_path in inputs.items():
        result[name] = audit(
            input_path,
            args.output.with_name(f"{name}_vector_audit.json"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
