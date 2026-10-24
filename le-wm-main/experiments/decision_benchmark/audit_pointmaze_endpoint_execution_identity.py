"""Independently bind PointMaze P1 outcome terminals to candidate identities.

The endpoint runner executes outcome rows in context-file order, whereas the
P1 diagnostic consumes root-local arrays.  This audit replays the sealed
outcome actions and proves that every terminal, cost, and root-local array is
still attached to the same ``(instance_id, candidate_id)`` identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _fresh_terminal


SCHEMA = "pointmaze_endpoint_execution_identity_audit.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite terminal [4]")
    return array


def _action(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape != (20, 2) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite action sequence [20,2]")
    return array


def audit(
    *, report_path: Path, output: Path,
    replay: Callable[[list[dict[str, Any]]], np.ndarray] = _fresh_terminal,
) -> dict[str, Any]:
    """Replay sealed outcome actions and compare every persisted identity view."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = _load(report_path)
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("endpoint report lacks context provenance")
    context_path = Path(str(provenance.get("context", "")))
    if not context_path.is_file() or _sha256(context_path) != provenance.get("context_sha256"):
        raise ValueError("endpoint context is missing or hash-mismatched")
    context = _load(context_path)
    rows = [dict(row) for row in context.get("records", []) if str(row.get("split")) == "outcome"]
    if not rows:
        raise ValueError("endpoint context contains no outcome rows")
    identities = [(str(row.get("instance_id", "")), str(row.get("candidate_id", ""))) for row in rows]
    if any(not root or not candidate for root, candidate in identities) or len(set(identities)) != len(identities):
        raise ValueError("outcome context has missing or duplicate root/candidate identities")
    replayed = np.asarray(replay(rows), dtype=np.float64)
    if replayed.shape != (len(rows), 4) or not np.isfinite(replayed).all():
        raise ValueError("independent outcome replay returned incompatible terminal vectors")

    labels = report.get("post_execution_outcome_labels")
    outcome = report.get("outcome", {}).get("per_root")
    binding = report.get("pre_outcome_prediction_bundle")
    if not isinstance(labels, dict) or not isinstance(outcome, dict) or not isinstance(binding, dict):
        raise ValueError("endpoint report lacks outcome labels, per-root report, or sealed predictions")
    bundle_path = Path(str(binding.get("path", "")))
    if not bundle_path.is_file() or _sha256(bundle_path) != binding.get("sha256"):
        raise ValueError("sealed prediction bundle is missing or hash-mismatched")
    predictions = _load(bundle_path).get("outcome_predictions")
    if not isinstance(predictions, dict):
        raise ValueError("sealed prediction bundle lacks outcome predictions")

    by_root: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    maximum_terminal_error = maximum_cost_error = 0.0
    expected_keys: set[str] = set()
    for row, terminal in zip(rows, replayed):
        root, candidate = str(row["instance_id"]), str(row["candidate_id"])
        key = f"{root}/{candidate}"
        expected_keys.add(key)
        label = labels.get(key)
        if not isinstance(label, dict) or str(label.get("instance_id")) != root or str(label.get("candidate_id")) != candidate:
            raise ValueError(f"{key}: post-execution label identity is missing or mismatched")
        recorded_terminal = _vector(label.get("realized_terminal_state"), name=f"{key} label")
        error = float(np.max(np.abs(recorded_terminal - terminal)))
        maximum_terminal_error = max(maximum_terminal_error, error)
        if error > 1.0e-10:
            raise ValueError(f"{key}: post-execution terminal does not match independent replay")
        goal = np.asarray(row.get("goal_raw"), dtype=np.float64)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError(f"{key}: context goal is invalid")
        cost_error = abs(float(label.get("real_task_cost", np.nan)) - float(np.linalg.norm(terminal[:2] - goal)))
        maximum_cost_error = max(maximum_cost_error, cost_error)
        if not np.isfinite(cost_error) or cost_error > 1.0e-10:
            raise ValueError(f"{key}: post-execution cost does not match independent replay")
        by_root[root].append((row, terminal))
    if set(labels) != expected_keys:
        raise ValueError("post-execution label identities differ from the sealed outcome context")
    if set(outcome) != set(by_root) or set(predictions) != set(by_root):
        raise ValueError("outcome root identities differ across context, report, and sealed predictions")

    for root, pairs in by_root.items():
        context_ids = [str(row["candidate_id"]) for row, _terminal in pairs]
        report_row, prediction = outcome[root], predictions[root]
        report_ids = [str(value) for value in report_row.get("candidate_ids", [])]
        prediction_ids = [str(value) for value in prediction.get("candidate_ids", [])]
        if context_ids != report_ids or context_ids != prediction_ids:
            raise ValueError(f"{root}: candidate ordering differs across context, report, or sealed predictions")
        context_actions = [_action(row["action_sequence"], name=f"{root}/{candidate} context") for row, candidate in zip((row for row, _ in pairs), context_ids)]
        prediction_actions = [_action(action, name=f"{root} prediction") for action in prediction.get("candidate_action_sequences", [])]
        if len(prediction_actions) != len(context_actions) or any(not np.array_equal(left, right) for left, right in zip(context_actions, prediction_actions)):
            raise ValueError(f"{root}: sealed action sequences differ from the executed outcome context")
        expected_terminal = np.stack([terminal for _row, terminal in pairs])
        reported_terminal = np.asarray(report_row.get("realized_candidate_terminal_state"), dtype=np.float64)
        if reported_terminal.shape != expected_terminal.shape or not np.allclose(reported_terminal, expected_terminal, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{root}: root-local terminal array does not preserve candidate ordering")
        expected_cost = np.asarray([np.linalg.norm(terminal[:2] - np.asarray(row["goal_raw"], dtype=np.float64)) for row, terminal in pairs])
        reported_cost = np.asarray(report_row.get("realized_candidate_cost"), dtype=np.float64)
        if reported_cost.shape != expected_cost.shape or not np.allclose(reported_cost, expected_cost, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{root}: root-local realized costs do not preserve candidate ordering")

    result = {
        "schema": SCHEMA,
        "status": "passed_independent_outcome_identity_replay",
        "report": str(report_path.resolve()),
        "context_provenance": {"path": str(context_path.resolve()), "sha256": provenance["context_sha256"]},
        "outcome_root_count": len(by_root),
        "outcome_candidate_count": len(rows),
        "maximum_terminal_absolute_error": maximum_terminal_error,
        "maximum_cost_absolute_error": maximum_cost_error,
        "claim_boundary": "This validates PointMaze outcome identity/order and deterministic terminal replay only. It does not establish predictive accuracy, spectral-regret association, or repair value.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(report_path=args.report, output=args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
