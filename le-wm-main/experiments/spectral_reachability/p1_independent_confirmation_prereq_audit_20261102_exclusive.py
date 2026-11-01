"""Fail-closed audit of the P1 independent-confirmation prerequisites.

This tool consumes an explicit metadata manifest from stdin.  It never imports
the model, dataset, simulator, CUDA, or any experiment runner.  A declaration
is not upgraded into evidence: every required field must be authoritative and
``verified`` before the contract can pass.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from typing import Any

SCHEMA = "p1_independent_confirmation_prereq_manifest.v1"
REPORT_SCHEMA = SCHEMA + ".report"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 2 * 1024 * 1024

REQUIRED_TOP = {
    "schema",
    "candidate",
    "root_identity",
    "source_receipt",
    "historical_dedup",
    "checkpoint_provenance",
    "negative_controls",
}


def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha(value: Any, field: str) -> None:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")


def _status(value: Any, field: str) -> None:
    if value not in {"verified", "unknown", "contradicted"}:
        raise ValueError(f"{field} must be verified, unknown, or contradicted")


def _exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{field} has an unexpected field set")
    return value


def _nonempty_text(value: Any, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty text")


def audit(manifest: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(manifest, REQUIRED_TOP, "manifest")
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected manifest schema")

    candidate = _exact_keys(
        manifest["candidate"],
        {"candidate_id", "role", "source_namespace"},
        "candidate",
    )
    for key in ("candidate_id", "role", "source_namespace"):
        _nonempty_text(candidate[key], f"candidate.{key}")

    root = _exact_keys(
        manifest["root_identity"],
        {
            "status",
            "root_id",
            "reset_seed",
            "root_sha256",
            "goal_sha256",
            "action_history_sha256",
            "authority",
            "historical_scope",
        },
        "root_identity",
    )
    _status(root["status"], "root_identity.status")
    _nonempty_text(root["root_id"], "root_identity.root_id")
    if type(root["reset_seed"]) is not int or root["reset_seed"] < 0:
        raise ValueError("root_identity.reset_seed must be a nonnegative integer")
    for key in ("root_sha256", "goal_sha256", "action_history_sha256"):
        _sha(root[key], f"root_identity.{key}")
    for key in ("authority", "historical_scope"):
        _nonempty_text(root[key], f"root_identity.{key}")

    receipt = _exact_keys(
        manifest["source_receipt"],
        {
            "status",
            "receipt_sha256",
            "queue_sha256",
            "reset_event_sha256",
            "producer_run_id",
            "complete",
            "files",
        },
        "source_receipt",
    )
    _status(receipt["status"], "source_receipt.status")
    for key in ("receipt_sha256", "queue_sha256", "reset_event_sha256"):
        _sha(receipt[key], f"source_receipt.{key}")
    _nonempty_text(receipt["producer_run_id"], "source_receipt.producer_run_id")
    if type(receipt["complete"]) is not bool:
        raise ValueError("source_receipt.complete must be boolean")
    if type(receipt["files"]) is not list or not receipt["files"]:
        raise ValueError("source_receipt.files must be non-empty")
    for index, item in enumerate(receipt["files"]):
        item = _exact_keys(item, {"path", "sha256", "role"}, f"source_receipt.files[{index}]")
        _nonempty_text(item["path"], f"source_receipt.files[{index}].path")
        _sha(item["sha256"], f"source_receipt.files[{index}].sha256")
        _nonempty_text(item["role"], f"source_receipt.files[{index}].role")

    dedup = _exact_keys(
        manifest["historical_dedup"],
        {
            "status",
            "index_sha256",
            "scope",
            "root_sha256_checked",
            "goal_sha256_checked",
            "action_history_sha256_checked",
            "exact_duplicate_count",
            "approximate_duplicate_check",
            "complete",
        },
        "historical_dedup",
    )
    _status(dedup["status"], "historical_dedup.status")
    _sha(dedup["index_sha256"], "historical_dedup.index_sha256")
    _nonempty_text(dedup["scope"], "historical_dedup.scope")
    for key in (
        "root_sha256_checked",
        "goal_sha256_checked",
        "action_history_sha256_checked",
    ):
        _sha(dedup[key], f"historical_dedup.{key}")
    if type(dedup["exact_duplicate_count"]) is not int or dedup["exact_duplicate_count"] < 0:
        raise ValueError("historical_dedup.exact_duplicate_count must be nonnegative")
    if type(dedup["approximate_duplicate_check"]) is not bool:
        raise ValueError("historical_dedup.approximate_duplicate_check must be boolean")
    if type(dedup["complete"]) is not bool:
        raise ValueError("historical_dedup.complete must be boolean")

    checkpoint = _exact_keys(
        manifest["checkpoint_provenance"],
        {
            "status",
            "checkpoint_sha256",
            "config_sha256",
            "training_run_id",
            "training_dataset_sha256",
            "training_split_manifest_sha256",
            "normalizer_sha256",
            "training_code_revision_sha256",
            "split_membership",
            "authoritative",
        },
        "checkpoint_provenance",
    )
    _status(checkpoint["status"], "checkpoint_provenance.status")
    for key in (
        "checkpoint_sha256",
        "config_sha256",
        "training_dataset_sha256",
        "training_split_manifest_sha256",
        "normalizer_sha256",
        "training_code_revision_sha256",
    ):
        _sha(checkpoint[key], f"checkpoint_provenance.{key}")
    _nonempty_text(checkpoint["training_run_id"], "checkpoint_provenance.training_run_id")
    _nonempty_text(checkpoint["split_membership"], "checkpoint_provenance.split_membership")
    if type(checkpoint["authoritative"]) is not bool:
        raise ValueError("checkpoint_provenance.authoritative must be boolean")

    controls = _exact_keys(
        manifest["negative_controls"],
        {"dev303_used", "gate_modified", "outcome_used_for_selection"},
        "negative_controls",
    )
    for key in controls:
        if type(controls[key]) is not bool:
            raise ValueError(f"negative_controls.{key} must be boolean")

    gaps: list[str] = []
    if candidate["source_namespace"].lower().startswith("dev303") or "dev303" in candidate["candidate_id"].lower():
        gaps.append("candidate_is_dev303_and_cannot_be_independent_confirmation")
    if root["status"] != "verified" or root["authority"] != "independent_authority":
        gaps.append("root_identity_not_independently_authoritative")
    if receipt["status"] != "verified" or not receipt["complete"]:
        gaps.append("source_receipt_incomplete_or_unverified")
    if dedup["status"] != "verified" or not dedup["complete"]:
        gaps.append("historical_dedup_incomplete_or_unverified")
    if not dedup["approximate_duplicate_check"]:
        gaps.append("approximate_or_semantic_history_dedup_missing")
    if checkpoint["status"] != "verified" or not checkpoint["authoritative"]:
        gaps.append("checkpoint_training_split_provenance_unverified")
    if checkpoint["split_membership"] != "training_disjoint":
        gaps.append("checkpoint_training_split_membership_not_proven_disjoint")
    for key, label in (
        ("dev303_used", "dev303_result_used"),
        ("gate_modified", "gate_modified"),
        ("outcome_used_for_selection", "outcome_used_for_selection"),
    ):
        if controls[key]:
            gaps.append(f"negative_control_{label}")

    passed = not gaps
    return {
        "schema": REPORT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "verdict": "PASS" if passed else "FAIL_CLOSED",
        "P1_independent_confirmation_authorized": passed,
        "gaps": gaps,
        "required_evidence": {
            "root_identity": "verified + independent_authority",
            "source_receipt": "verified + complete",
            "historical_dedup": "verified + complete + approximate_duplicate_check=true",
            "checkpoint_provenance": "verified + authoritative + training_disjoint",
            "negative_controls": "all false",
        },
        "operations": {
            "gpu": False,
            "physics": False,
            "training": False,
            "gate_mutation": False,
            "dev303_result_consumption": False,
        },
    }


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("manifest exceeds size cap")
        manifest = json.loads(
            raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
        report = audit(manifest)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0 if report["verdict"] == "PASS" else 2
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "verdict": "FAIL_CLOSED", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
