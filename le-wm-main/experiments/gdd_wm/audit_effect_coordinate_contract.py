"""Audit the action/effect coordinate contract used by GDD-WM.

This is a read-only audit.  It does not fit a teacher, read outcome labels for
selection, or alter any candidate package.  It resolves each terminal state
through the original candidate-bank artifact and checks whether the observed
XY displacement is expressed in the same chart as the action sequence.

The audit is deliberately weaker than a causal-identifiability claim: a
negative action/effect cosine can be caused by inertia, wall contacts, or
non-unique action-to-force dynamics.  We therefore report several invariant
and chart-transformed diagnostics instead of silently repairing signs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected a candidate dataset with records")
    return payload


def _array_key(reference: str) -> str:
    marker = "arrays.npz::"
    if marker not in str(reference):
        raise ValueError(f"unresolvable array reference: {reference}")
    return str(reference).split(marker, 1)[1]


def _dataset_with_arrays(
    dataset: Path,
    row: dict[str, Any],
    visited: set[Path],
) -> Path:
    dataset = dataset.resolve()
    if dataset in visited:
        raise ValueError(f"cyclic candidate-dataset provenance at {dataset}")
    visited.add(dataset)
    arrays_path = dataset.parent / "arrays.npz"
    if arrays_path.exists():
        return dataset
    payload = _load(dataset)
    source_candidate_id = str(row.get("source_candidate_id", row.get("candidate_id")))
    matches = [
        candidate for candidate in payload["records"]
        if str(candidate.get("instance_id")) == str(row.get("instance_id"))
        and str(candidate.get("candidate_id")) in {
            str(row.get("candidate_id")), source_candidate_id,
        }
    ]
    if not matches:
        raise FileNotFoundError(
            f"{dataset}: no provenance row for {row.get('instance_id')} / {row.get('candidate_id')}"
        )
    next_path = Path(str(matches[0].get("source_candidate_dataset", "")))
    if not next_path:
        raise FileNotFoundError(f"{dataset}: no source_candidate_dataset provenance")
    if not next_path.is_absolute():
        next_path = (Path.cwd() / next_path).resolve()
    return _dataset_with_arrays(next_path, row, visited)


def _resolve_realized(row: dict[str, Any], dataset_path: Path, cache: dict[Path, Any]) -> np.ndarray:
    source = Path(str(row.get("source_candidate_dataset", dataset_path)))
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    source = _dataset_with_arrays(source, row, set())
    arrays_path = source.parent / "arrays.npz"
    if arrays_path not in cache:
        cache[arrays_path] = np.load(arrays_path, allow_pickle=False)
    arrays = cache[arrays_path]
    key = _array_key(str(row["realized_terminal_state_ref"]))
    if key not in arrays.files:
        raise ValueError(f"{arrays_path}: missing {key}")
    value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if value.size < 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid terminal state for {row.get('instance_id')} / {key}")
    return value


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else float("nan")


def _summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None, "positive_fraction": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "positive_fraction": float(np.mean(finite > 0.0)),
    }


def audit(dataset_path: Path, output: Path) -> dict[str, Any]:
    payload = _load(dataset_path)
    records = payload["records"]
    if not records:
        raise ValueError("candidate dataset is empty")
    metadata = dict(payload.get("metadata", {}))
    cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    for row in records:
        source = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
        sequence = np.asarray(row["action_sequence"], dtype=np.float64)
        if source.size < 2 or sequence.ndim != 2 or sequence.shape[1] != 2:
            raise ValueError(f"invalid state/action shape for {row.get('instance_id')}")
        realized = _resolve_realized(row, dataset_path, cache)
        displacement = realized[:2] - source[:2]
        first = np.mean(sequence[: len(sequence) // 2], axis=0)
        second = np.mean(sequence[len(sequence) // 2 :], axis=0)
        cumulative = np.mean(sequence, axis=0)
        goal_delta = np.asarray(row["goal_raw"], dtype=np.float64)[:2] - source[:2]
        rows.append({
            "instance_id": str(row["instance_id"]),
            "candidate_id": str(row["candidate_id"]),
            "split": str(row.get("split")),
            "source_candidate_dataset": str(row.get("source_candidate_dataset")),
            "source_xy": source[:2].tolist(),
            "realized_xy": realized[:2].tolist(),
            "observed_displacement_xy": displacement.tolist(),
            "first_half_action_xy": first.tolist(),
            "second_half_action_xy": second.tolist(),
            "mean_action_xy": cumulative.tolist(),
            "goal_delta_xy": goal_delta.tolist(),
            "cosine_mean_action_to_effect": _cosine(cumulative, displacement),
            "cosine_first_action_to_effect": _cosine(first, displacement),
            "cosine_second_action_to_effect": _cosine(second, displacement),
            "cosine_goal_delta_to_effect": _cosine(goal_delta, displacement),
            "effect_norm": float(np.linalg.norm(displacement)),
        })

    # A moving PointMaze state has substantial inertial displacement even under
    # a near-zero pulse.  The raw action/effect cosine can therefore be
    # negative without any chart error.  Estimate a within-root no-action
    # baseline and report the residual separately; this is still diagnostic,
    # not an action-effect causal estimate.
    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_root.setdefault(row["instance_id"], []).append(row)
    for root_rows in by_root.values():
        baseline_row = min(root_rows, key=lambda item: float(np.linalg.norm(item["mean_action_xy"])))
        baseline = np.asarray(baseline_row["observed_displacement_xy"], dtype=np.float64)
        for row in root_rows:
            residual = np.asarray(row["observed_displacement_xy"], dtype=np.float64) - baseline
            row["within_root_baseline_candidate_id"] = baseline_row["candidate_id"]
            row["within_root_baseline_displacement_xy"] = baseline.tolist()
            row["baseline_corrected_displacement_xy"] = residual.tolist()
            row["cosine_mean_action_to_baseline_corrected_effect"] = _cosine(
                np.asarray(row["mean_action_xy"], dtype=np.float64), residual,
            )

    effects = np.asarray([r["observed_displacement_xy"] for r in rows], dtype=np.float64)
    actions = np.asarray([r["mean_action_xy"] for r in rows], dtype=np.float64)
    # Candidate-bank action charts may have a global axis reflection or swap.
    # Report these as diagnostics only; no transformed chart is selected here.
    transforms = {
        "identity": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "flip_x": np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        "flip_y": np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        "flip_xy": np.asarray([[-1.0, 0.0], [0.0, -1.0]]),
        "swap_xy": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "swap_flip_x": np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        "swap_flip_y": np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        "swap_flip_xy": np.asarray([[0.0, -1.0], [-1.0, 0.0]]),
    }
    chart_scores = {}
    for name, matrix in transforms.items():
        transformed = actions @ matrix.T
        chart_scores[name] = _summary(np.asarray([_cosine(a, e) for a, e in zip(transformed, effects)]))

    split_summary: dict[str, Any] = {}
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        split_summary[split] = {
            "rows": len(selected),
            "mean_action_effect": _summary(np.asarray([r["cosine_mean_action_to_effect"] for r in selected])),
            "first_action_effect": _summary(np.asarray([r["cosine_first_action_to_effect"] for r in selected])),
            "second_action_effect": _summary(np.asarray([r["cosine_second_action_to_effect"] for r in selected])),
            "goal_effect": _summary(np.asarray([r["cosine_goal_delta_to_effect"] for r in selected])),
            "effect_norm_mean": float(np.mean([r["effect_norm"] for r in selected])),
            "baseline_corrected_action_effect": _summary(np.asarray([
                r["cosine_mean_action_to_baseline_corrected_effect"] for r in selected
            ])),
        }
    result = {
        "schema": "gdd_wm_effect_coordinate_contract_audit.v0.1",
        "status": "completed",
        "protocol": {
            "dataset": str(dataset_path.resolve()),
            "records": len(rows),
            "outcome_labels_used_for_selection": False,
            "teacher_or_planner_fitted": False,
            "terminal_state_resolution": "source_candidate_dataset/arrays.npz",
            "action_chart_declared_by_metadata": metadata.get("candidate_bank_coordinate_frame"),
        },
        "split_summary": split_summary,
        "global_chart_transform_diagnostics": chart_scores,
        "row_count": len(rows),
        "rows": rows,
        "interpretation": {
            "coordinate_mismatch_evidence": bool(
                chart_scores["identity"]["mean"] is not None
                and chart_scores["identity"]["mean"] < 0.0
                and max((v["mean"] or -np.inf) for v in chart_scores.values()) > 0.0
            ),
            "claim_boundary": "coordinate-contract diagnostic; negative cosine is not by itself proof of a sign bug",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.output)
    print(json.dumps({
        "status": result["status"],
        "rows": result["row_count"],
        "split_summary": result["split_summary"],
        "global_chart_transform_diagnostics": result["global_chart_transform_diagnostics"],
        "interpretation": result["interpretation"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
