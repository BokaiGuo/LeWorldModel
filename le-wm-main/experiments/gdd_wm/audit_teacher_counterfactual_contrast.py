"""Diagnose why a frozen PointMaze teacher fails to change real decisions.

This is a post-execution mechanism audit.  Given an immutable candidate
dataset whose rows already contain realized terminal states and task costs, it
separates three failure modes *within each root/context candidate set*:

``predicted_action_effect_collapse``
    The decoded predicted endpoints do not retain enough candidate variation.
``predicted_dynamics_mismatch``
    Predicted latent candidate variation itself disagrees with realized
    terminal-state geometry.
``endpoint_readout_geometry_mismatch``
    Latent variation is preserved, but the fixed latent-to-endpoint readout
    fails to decode that variation into task-relevant XY geometry.
``task_readout_mismatch``
    Candidate endpoint geometry agrees, but the fixed teacher score still
    fails to order realized task cost.

The audit never refits a teacher, chooses a threshold, or authorizes a
planner.  It is deliberately post-execution evidence used to decide which
*future, separately frozen* repair family is scientifically warranted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_teacher_counterfactual_contrast_audit.v0.1"
_EPS = 1.0e-12


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected an object with metadata and records")
    if not payload["records"]:
        raise ValueError(f"{path}: records must be non-empty")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_from_reference(
    row: dict[str, Any],
    field: str,
    cache: dict[Path, Any],
) -> np.ndarray:
    reference = str(row.get(field, ""))
    source_dataset = row.get("predicted_latent_source_dataset")
    if "arrays.npz::" not in reference or not isinstance(source_dataset, str) or not source_dataset:
        raise ValueError(f"{row.get('instance_id')}/{row.get('candidate_id')}: {field} needs arrays.npz reference and source dataset")
    key = reference.split("arrays.npz::", 1)[1]
    dataset_path = Path(source_dataset).expanduser().resolve()
    array_paths = [dataset_path.parent / "arrays.npz"]
    # The shared-action-anchor protocol copies predicted arrays into each
    # context, but deliberately leaves large realized-state arrays immutable
    # in its declared source dataset.  The protocol report is the only allowed
    # fallback provenance edge; guessing among nearby result directories would
    # silently turn this diagnosis into an untraceable data join.
    protocol = dataset_path.parents[2] / "protocol_report.json" if len(dataset_path.parents) >= 3 else None
    if field == "realized_terminal_state_ref" and protocol is not None and protocol.exists():
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{protocol}: protocol report must be a JSON object")
        source = payload.get("source_candidate_dataset")
        if isinstance(source, str) and source:
            array_paths.append(Path(source).expanduser().resolve().parent / "arrays.npz")
    attempted: list[str] = []
    for arrays_path in array_paths:
        attempted.append(str(arrays_path))
        if not arrays_path.exists():
            continue
        if arrays_path not in cache:
            cache[arrays_path] = np.load(arrays_path, allow_pickle=False)
        arrays = cache[arrays_path]
        if key not in arrays.files:
            continue
        value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"{arrays_path}: non-finite {field} for {key}")
        return value
    raise ValueError(f"{dataset_path}: unable to resolve {field} key {key}; tried {attempted}")


def _pairwise_upper(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("pairwise comparison requires a finite [candidate, feature] array")
    if not np.isfinite(values).all():
        raise ValueError("pairwise comparison values must be finite")
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    return distances[np.triu_indices(len(values), k=1)]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("correlation operands must have equal shape")
    if len(left) < 2 or np.std(left) <= _EPS or np.std(right) <= _EPS:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((values - values.mean(axis=0, keepdims=True)) ** 2, axis=1))))


def _task_pair_agreement(score: np.ndarray, cost: np.ndarray, margin: float) -> tuple[int, int, float | None]:
    correct = comparable = 0
    for left in range(len(score)):
        for right in range(left + 1, len(score)):
            # Higher score and lower realized cost are both preferred.
            score_delta = float(score[left] - score[right])
            cost_delta = float(cost[right] - cost[left])
            if abs(score_delta) <= margin or abs(cost_delta) <= margin:
                continue
            comparable += 1
            correct += int((score_delta > 0.0) == (cost_delta > 0.0))
    return correct, comparable, None if comparable == 0 else float(correct / comparable)


def _failure_mode(
    *,
    spread_ratio: float,
    endpoint_correlation: float | None,
    latent_correlation: float | None,
    task_agreement: float | None,
    spread_ratio_threshold: float,
    variation_correlation_threshold: float,
    task_agreement_threshold: float,
) -> str:
    if spread_ratio < spread_ratio_threshold:
        return "predicted_action_effect_collapse"
    if latent_correlation is None or latent_correlation < variation_correlation_threshold:
        return "predicted_dynamics_mismatch"
    if endpoint_correlation is None or endpoint_correlation < variation_correlation_threshold:
        return "endpoint_readout_geometry_mismatch"
    if task_agreement is None or task_agreement <= task_agreement_threshold:
        return "task_readout_mismatch"
    return "no_dominant_failure_identified"


def _decision_group_report(
    rows: list[dict[str, Any]],
    cache: dict[Path, Any],
    *,
    score_margin: float,
    spread_ratio_threshold: float,
    variation_correlation_threshold: float,
    task_agreement_threshold: float,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["candidate_id"]))
    if len(ordered) < 3:
        raise ValueError("every decision group needs at least three candidate rows for contrast diagnosis")
    root = str(ordered[0]["instance_id"])
    set_id = str(ordered[0]["teacher_set_id"])
    if any(str(row["instance_id"]) != root or str(row["teacher_set_id"]) != set_id for row in ordered):
        raise ValueError("decision group mixes root or teacher set identities")
    predicted_latent = np.stack([_array_from_reference(row, "predicted_terminal_latent_ref", cache) for row in ordered])
    realized_state = np.stack([_array_from_reference(row, "realized_terminal_state_ref", cache) for row in ordered])
    if realized_state.shape[1] < 2:
        raise ValueError(f"{root}/{set_id}: realized terminal state lacks XY")
    predicted_xy = np.asarray([row.get("teacher_predicted_terminal_xy") for row in ordered], dtype=np.float64)
    if predicted_xy.shape != (len(ordered), 2) or not np.isfinite(predicted_xy).all():
        raise ValueError(f"{root}/{set_id}: teacher_predicted_terminal_xy must be finite [candidate,2]")
    realized_xy = realized_state[:, :2]
    scores = np.asarray([float(row["teacher_score"]) for row in ordered], dtype=np.float64)
    costs = np.asarray([float(row["real_task_cost"]) for row in ordered], dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(costs).all():
        raise ValueError(f"{root}/{set_id}: teacher scores and realized costs must be finite")

    predicted_xy_pairwise = _pairwise_upper(predicted_xy)
    realized_xy_pairwise = _pairwise_upper(realized_xy)
    predicted_latent_pairwise = _pairwise_upper(predicted_latent)
    # State is intentionally not lifted into latent space: this second
    # correlation asks only whether the frozen latent retains action-relative
    # variation in any monotone geometry, while the endpoint correlation asks
    # whether the fixed task-relevant XY readout decodes it correctly.
    endpoint_correlation = _safe_correlation(predicted_xy_pairwise, realized_xy_pairwise)
    latent_correlation = _safe_correlation(predicted_latent_pairwise, realized_xy_pairwise)
    correct, comparable, task_agreement = _task_pair_agreement(scores, costs, score_margin)
    predicted_spread = _spread(predicted_xy)
    realized_spread = _spread(realized_xy)
    spread_ratio = float(predicted_spread / max(realized_spread, _EPS))
    mode = _failure_mode(
        spread_ratio=spread_ratio,
        endpoint_correlation=endpoint_correlation,
        latent_correlation=latent_correlation,
        task_agreement=task_agreement,
        spread_ratio_threshold=spread_ratio_threshold,
        variation_correlation_threshold=variation_correlation_threshold,
        task_agreement_threshold=task_agreement_threshold,
    )
    return {
        "instance_id": root,
        "teacher_set_id": set_id,
        "split": str(ordered[0]["split"]),
        "candidate_count": len(ordered),
        "candidate_ids": [str(row["candidate_id"]) for row in ordered],
        "predicted_endpoint_xy_spread": predicted_spread,
        "realized_endpoint_xy_spread": realized_spread,
        "predicted_to_realized_spread_ratio": spread_ratio,
        "predicted_vs_realized_endpoint_pairwise_correlation": endpoint_correlation,
        "predicted_latent_vs_realized_endpoint_pairwise_correlation": latent_correlation,
        "task_pair_correct_count": correct,
        "task_pair_comparable_count": comparable,
        "teacher_task_pair_agreement": task_agreement,
        "dominant_failure_mode": mode,
    }


def audit(
    dataset_path: Path,
    *,
    score_margin: float = 1.0e-3,
    spread_ratio_threshold: float = 0.25,
    variation_correlation_threshold: float = 0.30,
    task_agreement_threshold: float = 0.50,
) -> dict[str, Any]:
    """Run a read-only counterfactual contrast audit over all recorded splits."""

    if score_margin < 0.0 or spread_ratio_threshold <= 0.0 or not -1.0 <= variation_correlation_threshold <= 1.0:
        raise ValueError("invalid contrast-audit thresholds")
    payload = _load(dataset_path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    required = {
        "instance_id", "teacher_set_id", "candidate_id", "split", "teacher_score", "real_task_cost",
        "teacher_predicted_terminal_xy", "predicted_terminal_latent_ref", "realized_terminal_state_ref",
        "predicted_latent_source_dataset",
    }
    for index, original in enumerate(payload["records"]):
        row = dict(original)
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"row {index} lacks required fields {missing}")
        groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(row)
    cache: dict[Path, Any] = {}
    per_group = [
        _decision_group_report(
            rows, cache,
            score_margin=score_margin,
            spread_ratio_threshold=spread_ratio_threshold,
            variation_correlation_threshold=variation_correlation_threshold,
            task_agreement_threshold=task_agreement_threshold,
        )
        for _key, rows in sorted(groups.items())
    ]
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row["split"]) for row in per_group}):
        values = [row for row in per_group if row["split"] == split]
        modes = Counter(str(row["dominant_failure_mode"]) for row in values)
        numeric = lambda key: [float(row[key]) for row in values if row[key] is not None]
        agreements = numeric("teacher_task_pair_agreement")
        by_split[split] = {
            "decision_group_count": len(values),
            "root_count": len({str(row["instance_id"]) for row in values}),
            "dominant_failure_mode_counts": dict(sorted(modes.items())),
            "mean_predicted_to_realized_spread_ratio": float(np.mean(numeric("predicted_to_realized_spread_ratio"))),
            "mean_endpoint_pairwise_correlation": None if not numeric("predicted_vs_realized_endpoint_pairwise_correlation") else float(np.mean(numeric("predicted_vs_realized_endpoint_pairwise_correlation"))),
            "mean_latent_pairwise_correlation": None if not numeric("predicted_latent_vs_realized_endpoint_pairwise_correlation") else float(np.mean(numeric("predicted_latent_vs_realized_endpoint_pairwise_correlation"))),
            "mean_teacher_task_pair_agreement": None if not agreements else float(np.mean(agreements)),
        }
    return {
        "schema": SCHEMA,
        "status": "completed_post_execution_mechanism_audit",
        "protocol": {
            "dataset": str(dataset_path.resolve()),
            "candidate_dataset_sha256": _sha256(dataset_path),
            "teacher_refit": False,
            "policy_selection_or_authorization": False,
            "realized_outcomes_used": "post_execution_diagnosis_only",
            "thresholds": {
                "score_margin": float(score_margin),
                "spread_ratio": float(spread_ratio_threshold),
                "variation_correlation": float(variation_correlation_threshold),
                "task_pair_agreement": float(task_agreement_threshold),
            },
        },
        "summary_by_split": by_split,
        "per_decision_group": per_group,
        "repair_interpretation": {
            "predicted_action_effect_collapse": "Future repair should change predictive action conditioning or collect discriminative probes; a task readout alone cannot restore absent candidate variation.",
            "predicted_dynamics_mismatch": "Future repair should target local verified dynamics or support-aware probing before changing the task geometry.",
            "endpoint_readout_geometry_mismatch": "Future repair should target the frozen latent-to-task-endpoint readout before changing the world-model dynamics.",
            "task_readout_mismatch": "Future repair may target goal-conditioned geometry/readout, but only with a separately frozen prospective protocol.",
            "no_dominant_failure_identified": "This taxonomy does not explain the group; do not infer a repair without a new diagnostic.",
        },
        "claim_boundary": "Post-execution root/context mechanism diagnosis only. It neither validates a teacher nor establishes distillation, planning, or closed-loop improvement.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score-margin", type=float, default=1.0e-3)
    parser.add_argument("--spread-ratio-threshold", type=float, default=0.25)
    parser.add_argument("--variation-correlation-threshold", type=float, default=0.30)
    parser.add_argument("--task-agreement-threshold", type=float, default=0.50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual contrast audit: {args.output}")
    report = audit(
        args.dataset,
        score_margin=args.score_margin,
        spread_ratio_threshold=args.spread_ratio_threshold,
        variation_correlation_threshold=args.variation_correlation_threshold,
        task_agreement_threshold=args.task_agreement_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary_by_split": report["summary_by_split"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
