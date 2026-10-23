"""Audit GDD-WM teacher preference stability and execution consistency.

The audit accepts the D0 shared-candidate schema, but it does not assume that
one teacher score per candidate proves stability.  Stability is estimable only
when the same root/candidate pair is observed under at least two distinct
candidate-set contexts.  Execution consistency is reported separately and
does not upgrade an offline cost oracle into a D-JEPA teacher.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}, [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object or list")
    rows = payload.get("records", payload.get("candidates"))
    if not isinstance(rows, list):
        raise ValueError("input object must contain records or candidates")
    return dict(payload.get("metadata", {})), rows


def _sign(value: float, margin: float) -> int:
    if abs(float(value)) <= float(margin):
        return 0
    return 1 if value > 0.0 else -1


def _candidate_identity(row: dict[str, Any]) -> str:
    """Return the executable action identity, never a context-local label.

    Shared-anchor protocols intentionally give the same executable action a
    distinct context-prefixed ``candidate_id``.  D1 stability is about whether
    that *action* retains its preference across candidate-set contexts, so
    using the local label makes every legitimate repeat invisible and forces a
    lossy post-hoc dataset rewrite.  A declared sequence hash is authoritative;
    historical inputs without one retain their candidate-id fallback.
    """
    action_hash = row.get("action_sequence_sha256")
    if isinstance(action_hash, str) and action_hash:
        return f"action_sequence_sha256::{action_hash}"
    return f"candidate_id::{row['candidate_id']}"


def _has_realized_successor_ref(row: dict[str, Any]) -> bool:
    """Accept either realized latent or realized physical terminal evidence."""
    return bool(row.get("realized_latent_ref") or row.get("realized_terminal_state_ref"))


def _pair_metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    teacher_consistent = 0
    teacher_comparable = 0
    teacher_inversions = 0
    pair_count = 0
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            pair_count += 1
            # Higher teacher score is preferred; lower execution cost is preferred.
            teacher_delta = float(rows[left]["teacher_score"]) - float(rows[right]["teacher_score"])
            real_delta = float(rows[right]["real_task_cost"]) - float(rows[left]["real_task_cost"])
            teacher_sign = _sign(teacher_delta, margin)
            real_sign = _sign(real_delta, margin)
            if teacher_sign == 0 or real_sign == 0:
                continue
            teacher_comparable += 1
            if teacher_sign == real_sign:
                teacher_consistent += 1
            else:
                teacher_inversions += 1
    return {
        "pair_count": int(pair_count),
        "comparable_pair_count": int(teacher_comparable),
        "execution_consistent_pair_count": int(teacher_consistent),
        "execution_inversion_count": int(teacher_inversions),
        "execution_consistency": None
        if teacher_comparable == 0
        else float(teacher_consistent / teacher_comparable),
    }


def _aggregate_group_metrics(groups: Iterable[list[dict[str, Any]]], margin: float) -> dict[str, Any]:
    """Aggregate pair counts over independent root/candidate-set groups."""

    totals = {
        "pair_count": 0,
        "comparable_pair_count": 0,
        "execution_consistent_pair_count": 0,
        "execution_inversion_count": 0,
    }
    for group_rows in groups:
        metrics = _pair_metrics(group_rows, margin)
        for key in totals:
            totals[key] += int(metrics[key])
    comparable = totals["comparable_pair_count"]
    totals["execution_consistency"] = None if comparable == 0 else float(
        totals["execution_consistent_pair_count"] / comparable
    )
    return totals


def audit_records(
    records: Iterable[dict[str, Any]],
    *,
    margin: float = 1.0e-3,
    require_realized_refs: bool = True,
) -> dict[str, Any]:
    """Return D1 diagnostics and an explicit eligibility decision."""

    rows = list(records)
    errors: list[str] = []
    warnings: list[str] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pair_contexts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    teacher_models: set[str] = set()
    generators: set[str] = set()
    missing_runtime_refs = 0

    for index, row in enumerate(rows):
        required = {"instance_id", "candidate_id", "teacher_set_id", "teacher_score", "real_task_cost", "split"}
        missing = sorted(required.difference(row)) if isinstance(row, dict) else ["record_object"]
        if missing:
            errors.append(f"row {index}: missing {missing}")
            continue
        try:
            float(row["teacher_score"])
            float(row["real_task_cost"])
        except (TypeError, ValueError):
            errors.append(f"row {index}: teacher_score and real_task_cost must be numeric")
            continue
        root = str(row["instance_id"])
        set_id = str(row["teacher_set_id"])
        groups[(root, set_id)].append(row)
        candidate = _candidate_identity(row)
        teacher_models.add(str(row.get("teacher_model_id", "missing")))
        generators.add(str(row.get("candidate_generator", "missing")))
        if row.get("split") in {"outcome", "environment_test"} and (
            not row.get("predicted_latent_ref") or not _has_realized_successor_ref(row)
        ):
            missing_runtime_refs += 1

    if not rows:
        errors.append("no records supplied")
    if float(margin) < 0.0 or not np.isfinite(float(margin)):
        errors.append("margin must be finite and non-negative")

    all_metrics = _aggregate_group_metrics(groups.values(), margin) if not errors else {}
    teacher_cost_residual = np.asarray(
        [float(row["teacher_score"]) + float(row["real_task_cost"]) for row in rows],
        dtype=np.float64,
    ) if rows and not errors else np.asarray([], dtype=np.float64)
    exact_negative_cost_match = bool(
        teacher_cost_residual.size
        and np.allclose(teacher_cost_residual, 0.0, rtol=0.0, atol=1.0e-10)
    )
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(row.get("split")) for row in rows if isinstance(row, dict)}):
        split_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if str(row.get("split")) == split:
                split_groups[(str(row["instance_id"]), str(row["teacher_set_id"]))].append(row)
        by_split[split] = _aggregate_group_metrics(split_groups.values(), margin)

    # A pair must appear with the same root and executable action identities in
    # at least two teacher-set contexts. Candidate IDs may legitimately be
    # context-local labels in a shared-anchor protocol.
    for (root, set_id), group_rows in groups.items():
        for left in range(len(group_rows)):
            for right in range(left + 1, len(group_rows)):
                left_row, right_row = group_rows[left], group_rows[right]
                left_id, right_id = sorted((_candidate_identity(left_row), _candidate_identity(right_row)))
                raw_delta = float(left_row["teacher_score"]) - float(right_row["teacher_score"])
                pair_contexts[(root, left_id, right_id)][set_id] = _sign(raw_delta, margin)
    repeated_contexts = [contexts for contexts in pair_contexts.values() if len(contexts) >= 2]
    if not repeated_contexts:
        warnings.append("no root/candidate pair has two candidate-set contexts; preference stability is unavailable")
        stability = {
            "status": "unavailable_insufficient_candidate_set_repeats",
            "pair_count_with_repeated_context": 0,
            "flip_rate": None,
            "stability_score": None,
        }
    else:
        signs = [sign for contexts in repeated_contexts for sign in contexts.values() if sign != 0]
        flips = sum(
            1
            for contexts in repeated_contexts
            if len({sign for sign in contexts.values() if sign != 0}) > 1
        )
        stability = {
            "status": "computed",
            "pair_count_with_repeated_context": len(repeated_contexts),
            "flip_rate": float(flips / len(repeated_contexts)),
            "stability_score": None if not signs else float(1.0 - (flips / len(repeated_contexts))),
        }

    if missing_runtime_refs:
        warnings.append(f"{missing_runtime_refs} outcome/test rows lack predicted or realized latent refs")
    if any(model.startswith("offline_environment_cost") for model in teacher_models):
        warnings.append("teacher model is an offline environment-cost oracle, not a validated D-JEPA teacher")
    if exact_negative_cost_match:
        warnings.append("teacher_score is exactly negative real_task_cost; teacher/execution agreement is label leakage")
    if len(teacher_models) != 1:
        warnings.append("teacher model identity is heterogeneous or missing")

    support_passed = not missing_runtime_refs if require_realized_refs else True
    independent_execution_consistency = not exact_negative_cost_match
    eligible = (
        not errors
        and stability["status"] == "computed"
        and support_passed
        and independent_execution_consistency
        and all_metrics.get("comparable_pair_count", 0) > 0
    )
    return {
        "schema": "gdd_wm_d1_preference_stability.v0.1",
        "status": "passed" if eligible else "rejected_fail_closed",
        "claim_boundary": "D1 diagnostics; teacher stability is not teacher correctness and does not establish planning",
        "summary": {
            "record_count": len(rows),
            "candidate_set_count": len(groups),
            "root_count": len({key[0] for key in groups}),
            "teacher_models": sorted(teacher_models),
            "candidate_generators": sorted(generators),
            "missing_runtime_ref_count": int(missing_runtime_refs),
            "require_realized_refs": bool(require_realized_refs),
        },
        "execution_consistency": {"all": all_metrics, "by_split": by_split},
        "teacher_execution_leakage": {
            "exact_negative_cost_match": exact_negative_cost_match,
            "residual_max_abs": None
            if not teacher_cost_residual.size
            else float(np.max(np.abs(teacher_cost_residual))),
            "residual_mean_abs": None
            if not teacher_cost_residual.size
            else float(np.mean(np.abs(teacher_cost_residual))),
            "independent_execution_consistency_valid": independent_execution_consistency,
        },
        "preference_stability": stability,
        "decision": {
            "d1_stability_estimable": stability["status"] == "computed",
            "d1_execution_consistency_estimable": all_metrics.get("comparable_pair_count", 0) > 0,
            "d1_independent_execution_consistency": independent_execution_consistency,
            "d1_eligible_pairs": "only_repeated_context_and_support_passed" if eligible else 0,
            "d2_ready": False,
            "scientific_readiness": "blocked_teacher_execution_validation",
        },
        "errors": errors,
        "warnings": warnings,
        "protocol": {
            "pair_margin": float(margin),
            "same_root_candidate_set_context_required": True,
            "execution_cost_direction": "lower_is_better",
            "teacher_score_direction": "higher_is_better",
            "runtime_teacher_forbidden": True,
            "outcome_used_to_select_teacher_or_threshold": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    parser.add_argument("--allow-missing-realized-refs", action="store_true")
    args = parser.parse_args()
    metadata, records = _load(args.input)
    report = audit_records(
        records,
        margin=args.margin,
        require_realized_refs=not args.allow_missing_realized_refs,
    )
    report["input_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
