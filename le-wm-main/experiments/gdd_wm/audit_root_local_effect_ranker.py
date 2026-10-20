"""Audit whether root-local observed effects repair GDD action ranking.

This is an execution-supervised diagnostic, not a frozen-teacher or closed-loop
claim.  For every root, a predeclared action-space coverage rule chooses K
probe candidates without reading task outcomes.  Their realized terminal
displacements fit a local ridge map from action basis to XY effect; the map
then ranks the remaining candidates.  The artifact reports this separately
from the frozen predictive teacher and action-only controls.
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

from experiments.decision_benchmark.train_candidate_effect_adapter import action_sequence_basis
from experiments.gdd_wm.audit_effect_coordinate_contract import _resolve_realized


def _ridge_fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ targets)


def _basis(row: dict[str, Any]) -> np.ndarray:
    return action_sequence_basis(np.asarray(row["action_sequence"], dtype=np.float64))


def _coverage_probe_indices(features: np.ndarray, count: int) -> list[int]:
    """Select action-space coverage points, independent of execution labels."""

    if count < 1 or count > len(features):
        raise ValueError("probe count must lie between one and candidate count")
    norms = np.linalg.norm(features, axis=1)
    selected = [int(np.argmin(norms))]
    while len(selected) < count:
        distances = np.min(
            np.sqrt(np.mean((features[:, None, :] - features[selected][None, :, :]) ** 2, axis=-1)),
            axis=1,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return selected


def _regret(rows: list[dict[str, Any]], selected_index: int, candidates: list[int]) -> float:
    selected = rows[selected_index]
    oracle = min(float(rows[index]["real_task_cost"]) for index in candidates)
    return float(float(selected["real_task_cost"]) - oracle)


def _root_bootstrap(values: list[float], seed: int = 20261019, draws: int = 5000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"estimate": None, "lower": None, "upper": None, "root_count": 0}
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        float(np.mean(array[rng.integers(0, len(array), len(array))]))
        for _ in range(draws)
    ])
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "root_count": int(len(array)),
    }


def audit(dataset: Path, output: Path, *, probe_count: int = 3, ridge: float = 1.0e-2) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("candidate dataset is empty")
    arrays_cache: dict[Path, Any] = {}
    enriched: list[dict[str, Any]] = []
    for row in records:
        source = np.asarray(row["source_state_raw"], dtype=np.float64)[:2]
        realized = _resolve_realized(row, dataset, arrays_cache)
        enriched.append({
            "row": row,
            "basis": _basis(row),
            "effect": realized[:2] - source,
            "goal": np.asarray(row["goal_raw"], dtype=np.float64)[:2],
        })
    by_root: dict[str, list[dict[str, Any]]] = {}
    for item in enriched:
        by_root.setdefault(str(item["row"]["instance_id"]), []).append(item)
    root_reports: list[dict[str, Any]] = []
    for root, items in sorted(by_root.items()):
        items = sorted(items, key=lambda item: str(item["row"]["candidate_id"]))
        if len(items) <= probe_count:
            continue
        bases = np.stack([item["basis"] for item in items])
        probes = _coverage_probe_indices(bases, probe_count)
        holdout = [index for index in range(len(items)) if index not in probes]
        probe_x = bases[probes]
        probe_y = np.stack([items[index]["effect"] for index in probes])
        beta = _ridge_fit(probe_x, probe_y, ridge)
        design = np.concatenate([np.ones((len(holdout), 1)), bases[holdout]], axis=1)
        predicted_effect = design @ beta
        source = np.asarray(items[0]["row"]["source_state_raw"], dtype=np.float64)[:2]
        goal = items[0]["goal"]
        predicted_cost = np.linalg.norm(source[None, :] + predicted_effect - goal[None, :], axis=1)
        local_choice = int(holdout[int(np.argmin(predicted_cost))])
        local_regret = _regret([item["row"] for item in items], local_choice, holdout)

        # Action-only control: fit mean terminal effect as an intercept from
        # the same probe observations and do not use candidate action geometry.
        mean_effect = np.mean(probe_y, axis=0)
        action_only_cost = np.linalg.norm(source + mean_effect - goal)
        action_only_choice = min(holdout, key=lambda index: float(np.linalg.norm(source + mean_effect - goal)))
        del action_only_cost  # retained choice is intentionally the intercept baseline
        action_only_regret = _regret([item["row"] for item in items], action_only_choice, holdout)

        teacher_candidates = [
            index for index in holdout
            if items[index]["row"].get("teacher_score") is not None
        ]
        teacher_choice = None
        teacher_regret = None
        if teacher_candidates:
            teacher_choice = min(
                teacher_candidates,
                key=lambda index: -float(items[index]["row"]["teacher_score"]),
            )
            teacher_regret = _regret([item["row"] for item in items], teacher_choice, holdout)
        root_reports.append({
            "instance_id": root,
            "split": str(items[0]["row"].get("split")),
            "candidate_count": len(items),
            "probe_count": len(probes),
            "probe_candidate_ids": [str(items[index]["row"]["candidate_id"]) for index in probes],
            "holdout_count": len(holdout),
            "local_effect_regret": local_regret,
            "probe_intercept_regret": action_only_regret,
            "frozen_teacher_regret": teacher_regret,
            "local_minus_intercept_regret": float(local_regret - action_only_regret),
            "local_minus_teacher_regret": None if teacher_regret is None else float(local_regret - teacher_regret),
        })

    if not root_reports:
        raise ValueError("no roots have enough candidates for the requested probe count")
    split_summary: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in root_reports}):
        subset = [row for row in root_reports if str(row["split"]) == split]
        split_summary[split] = {
            "local_effect_regret": _root_bootstrap([row["local_effect_regret"] for row in subset], 20261019),
            "probe_intercept_regret": _root_bootstrap([row["probe_intercept_regret"] for row in subset], 20261020),
            "local_minus_intercept_regret": _root_bootstrap([row["local_minus_intercept_regret"] for row in subset], 20261021),
            "local_minus_teacher_regret": _root_bootstrap([
                row["local_minus_teacher_regret"] for row in subset
                if row["local_minus_teacher_regret"] is not None
            ], 20261022),
        }
    result = {
        "schema": "gdd_wm_root_local_effect_ranker_audit.v0.1",
        "status": "completed_diagnostic",
        "protocol": {
            "dataset": str(dataset.resolve()),
            "probe_rule": "greedy farthest-point coverage in action basis; execution outcomes not read",
            "probe_count": int(probe_count),
            "ridge": float(ridge),
            "outcome_labels_used_for_probe_selection": False,
            "execution_supervised": True,
            "closed_loop_evidence": False,
        },
        "summary": {
            "root_count": len(root_reports),
            "mean_local_effect_regret": float(np.mean([r["local_effect_regret"] for r in root_reports])),
            "mean_probe_intercept_regret": float(np.mean([r["probe_intercept_regret"] for r in root_reports])),
            "mean_frozen_teacher_regret": (
                None if not any(r["frozen_teacher_regret"] is not None for r in root_reports)
                else float(np.mean([r["frozen_teacher_regret"] for r in root_reports if r["frozen_teacher_regret"] is not None]))
            ),
            "local_minus_intercept_mean": float(np.mean([r["local_minus_intercept_regret"] for r in root_reports])),
        },
        "split_summary": split_summary,
        "roots": root_reports,
        "claim_boundary": "execution-supervised root-local ranking diagnostic; no frozen teacher or closed-loop claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    args = parser.parse_args()
    result = audit(args.dataset, args.output, probe_count=args.probe_count, ridge=args.ridge)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
