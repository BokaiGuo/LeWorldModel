"""Validate and canonicalize the GDD-WM D0 shared candidate dataset.

This runner deliberately does not generate teacher labels or simulator
outcomes.  It turns an externally produced candidate table into an auditable
schema artifact and fails closed when provenance, support, or root-disjoint
split requirements are missing.  A schema-valid table is not evidence that a
D-JEPA teacher is execution-validated; that distinction is recorded in the
output report.

Accepted input formats are either a JSON object with ``records`` (or
``candidates``) and a ``metadata`` object, a JSON list of records, or JSONL.
The latter two formats are accepted for convenience but remain blocked unless
the required provenance metadata is supplied through ``--metadata``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = (
    "instance_id",
    "split",
    "environment",
    "start_episode_id",
    "goal_id",
    "candidate_id",
    "action_sequence",
    "candidate_generator",
    "cem_stage",
    "source_observation_ref",
    "goal_observation_ref",
    "teacher_model_id",
    "teacher_score",
    "teacher_rank",
    "teacher_set_id",
    "real_success",
    "real_task_cost",
    "realized_observation_ref",
    "predicted_latent_ref",
    "realized_latent_ref",
    "action_support",
    "privileged_state_role",
)
ALLOWED_SPLITS = frozenset({"train", "calibration", "outcome", "environment_test"})
REQUIRED_PROVENANCE = {
    "outcome_used_for_teacher_or_metric_selection": False,
    "snapshot_selection_from_outcome": False,
    "teacher_frozen_before_distillation": True,
    "shared_candidate_bank": True,
    "runtime_teacher_forbidden": True,
}


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return True


def _load_json_or_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {}, records
    if isinstance(payload, list):
        return {}, payload
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object, JSON list, or JSONL")
    records = payload.get("records", payload.get("candidates"))
    if not isinstance(records, list):
        raise ValueError("input object must contain a list under 'records' or 'candidates'")
    return dict(payload.get("metadata", {})), records


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metadata must be a JSON object")
    return payload


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["split"]), str(row["instance_id"]), str(row["teacher_set_id"]))


def validate_records(
    records: Iterable[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    expected_candidates: int = 32,
) -> dict[str, Any]:
    """Return an auditable D0 validation report without changing the input."""

    rows = list(records)
    errors: list[str] = []
    warnings: list[str] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    split_roots: dict[str, set[str]] = defaultdict(set)
    candidate_generators: set[str] = set()
    environments: set[str] = set()

    if not rows:
        errors.append("no candidate records supplied")
    if expected_candidates < 1:
        errors.append("expected_candidates must be positive")

    if not metadata:
        errors.append("provenance metadata missing")
    for key, expected in REQUIRED_PROVENANCE.items():
        if metadata.get(key) is not expected:
            errors.append(f"provenance.{key} must be {expected!r}")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: record is not an object")
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"row {index}: missing fields {missing}")
            continue
        split = str(row["split"])
        if split not in ALLOWED_SPLITS:
            errors.append(f"row {index}: unsupported split {split!r}")
        if not _finite_numbers(row["action_sequence"]):
            errors.append(f"row {index}: action_sequence contains non-finite values")
        if row["action_support"] in (None, "", [], {}):
            errors.append(f"row {index}: action_support is missing")
        if not isinstance(row["candidate_id"], (str, int)):
            errors.append(f"row {index}: candidate_id must be string or integer")
        if row["teacher_score"] is None or row["teacher_rank"] is None:
            errors.append(f"row {index}: teacher score/rank is missing")
        candidate_generators.add(str(row["candidate_generator"]))
        environments.add(str(row["environment"]))
        split_roots[split].add(str(row["start_episode_id"]))
        groups[_group_key(row)].append(row)

    for group, group_rows in sorted(groups.items()):
        candidate_ids = [str(row["candidate_id"]) for row in group_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append(f"group {group}: duplicate candidate_id")
        if len(group_rows) != expected_candidates:
            errors.append(
                f"group {group}: expected {expected_candidates} candidates, got {len(group_rows)}"
            )
        if len({str(row["environment"]) for row in group_rows}) != 1:
            errors.append(f"group {group}: environment is not constant")

    split_names = sorted(split_roots)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = sorted(split_roots[left] & split_roots[right])
            if overlap:
                errors.append(f"root-disjoint violation {left}/{right}: {overlap[:8]}")

    outcome_rows = sum(1 for row in rows if isinstance(row, dict) and row.get("split") == "outcome")
    if outcome_rows == 0:
        warnings.append("no outcome split: D1 execution consistency cannot be assessed")
    outcome_success_rows = sum(
        1
        for row in rows
        if isinstance(row, dict) and row.get("split") == "outcome" and bool(row.get("real_success", False))
    )
    outcome_groups_with_success = len({
        str(row["instance_id"])
        for row in rows
        if isinstance(row, dict) and row.get("split") == "outcome" and bool(row.get("real_success", False))
    })
    if outcome_rows and outcome_success_rows == 0:
        warnings.append("outcome split has zero positive task labels: success-based downstream estimands are unsupported")
    missing_runtime_refs = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("split") in {"outcome", "environment_test"}
        and (row.get("predicted_latent_ref") is None or row.get("realized_latent_ref") is None)
    )
    if missing_runtime_refs:
        warnings.append(f"{missing_runtime_refs} outcome/test rows lack predicted or realized latent refs")

    numeric_teacher_cost_pairs = [
        (float(row["teacher_score"]), float(row["real_task_cost"]))
        for row in rows
        if isinstance(row, dict)
        and row.get("teacher_score") is not None
        and row.get("real_task_cost") is not None
    ]
    exact_negative_cost_match = bool(numeric_teacher_cost_pairs) and all(
        abs(score + cost) <= 1.0e-10 for score, cost in numeric_teacher_cost_pairs
    )
    if exact_negative_cost_match:
        warnings.append(
            "teacher_score is exactly negative real_task_cost; D1 execution agreement would be label leakage"
        )

    schema_status = "passed" if not errors else "rejected_fail_closed"
    return {
        "schema": "gdd_wm_d0_candidate_dataset.v0.1",
        "status": schema_status,
        "claim_boundary": "schema_audit_only_not_teacher_validation",
        "metadata": metadata,
        "summary": {
            "record_count": len(rows),
            "candidate_set_count": len(groups),
            "expected_candidates_per_set": expected_candidates,
            "split_counts": {split: len(values) for split, values in sorted(split_roots.items())},
            "environment_count": len(environments),
            "environments": sorted(environments),
            "candidate_generators": sorted(candidate_generators),
            "outcome_record_count": outcome_rows,
            "outcome_success_record_count": outcome_success_rows,
            "outcome_groups_with_success": outcome_groups_with_success,
            "numeric_teacher_cost_pair_count": len(numeric_teacher_cost_pairs),
            "teacher_score_exact_negative_real_task_cost": exact_negative_cost_match,
        },
        "decision": {
            "d0_schema_complete": not bool(errors),
            "teacher_execution_validation": (
                "rejected_exact_negative_cost_leakage"
                if exact_negative_cost_match
                else "pending"
            ),
            "d1_ready": (
                not bool(errors)
                and outcome_rows > 0
                and missing_runtime_refs == 0
                and not exact_negative_cost_match
            ),
            "d2_ready": False,
            "scientific_readiness": "blocked_teacher_execution_validation",
        },
        "errors": errors,
        "warnings": warnings,
    }


def canonical_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic records for reproducible downstream joins."""

    return sorted(
        (dict(row) for row in records),
        key=lambda row: (
            str(row.get("split", "")),
            str(row.get("instance_id", "")),
            str(row.get("teacher_set_id", "")),
            str(row.get("candidate_id", "")),
        ),
    )


def build_artifact(
    input_path: Path,
    output_dir: Path,
    *,
    metadata_path: Path | None = None,
    expected_candidates: int = 32,
) -> dict[str, Any]:
    metadata, records = _load_json_or_jsonl(input_path)
    metadata.update(_load_metadata(metadata_path))
    report = validate_records(records, metadata, expected_candidates=expected_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "d0_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["status"] == "passed":
        with (output_dir / "candidate_records.jsonl").open("w", encoding="utf-8") as handle:
            for row in canonical_records(records):
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expected-candidates", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_artifact(
            args.input,
            args.output,
            metadata_path=args.metadata,
            expected_candidates=args.expected_candidates,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "gdd_wm_d0_candidate_dataset.v0.1",
            "status": "rejected_fail_closed",
            "claim_boundary": "schema_audit_only_not_teacher_validation",
            "decision": {"d0_schema_complete": False, "scientific_readiness": "input_error"},
            "errors": [str(exc)],
            "warnings": [],
        }
        (args.output / "d0_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
