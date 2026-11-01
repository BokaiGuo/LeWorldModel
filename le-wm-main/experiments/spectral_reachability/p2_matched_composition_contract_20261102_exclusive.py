"""CPU-only matched P2 composition comparison contract.

This module validates a sealed comparison description.  It does not import a
model/runtime, start CUDA, call physics, or alter any admission gate.
"""
from __future__ import annotations

import hashlib
import json
import math

NAME = "p2_matched_composition_contract_20261102_exclusive"
SCHEMA = NAME + ".v1"
ARMS = ("full_composition", "no_composition_sra", "zero", "strong_baseline")
RESOURCE_AXES = (
    "predict", "action_encoder", "forward_ad", "jacobian", "vjp",
    "svd", "bounded_solve", "backward", "wall_seconds",
)
STATUS = ("completed", "early_stop", "failed", "timeout", "blocked", "missing")
FAILURE_STATUSES = frozenset(("failed", "timeout", "blocked", "missing"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = value if isinstance(value, bytes) else value.encode()
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def _number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)
    return value


def _budget(row):
    budget = row.get("budget")
    require(isinstance(budget, dict) and tuple(budget) == RESOURCE_AXES,
            "fixed budget axes")
    for axis in RESOURCE_AXES:
        item = budget[axis]
        require(isinstance(item, dict)
                and tuple(item) == ("planned", "attempted", "completed", "failed",
                                    "pending", "unknown"), f"{axis} budget fields")
        planned = _number(item["planned"], f"{axis} planned")
        for key in ("attempted", "completed", "failed", "pending", "unknown"):
            _number(item[key], f"{axis} {key}")
        require(item["completed"] + item["failed"] <= item["attempted"],
                f"{axis} completed/failed exceeds attempted")
        require(item["attempted"] <= planned,
                f"{axis} attempted exceeds planned")
    return budget


def _row(row, root, arm):
    require(row.get("root") == root and row.get("arm") == arm, "fixed row identity")
    require(row.get("status") in STATUS, "row status")
    _budget(row)
    require(isinstance(row.get("failure"), dict), "failure record")
    failure = row["failure"]
    require(tuple(failure) == ("kind", "message", "retryable", "rerun_authorized"),
            "fixed failure fields")
    require(isinstance(failure["kind"], str) and isinstance(failure["message"], str)
            and type(failure["retryable"]) is bool
            and type(failure["rerun_authorized"]) is bool, "failure values")
    if row["status"] in FAILURE_STATUSES:
        require(failure["kind"] != "none" and failure["message"], "failure must be durable")
        require(failure["rerun_authorized"] is False, "no retry/reroute")
    else:
        require(failure["kind"] == "none" and failure["message"] == ""
                and failure["retryable"] is False
                and failure["rerun_authorized"] is False, "nonfailure record")
    require(row.get("outcome") is None, "outcomes are not opened by contract")


def build_contract(report):
    """Validate a complete matched comparison report and return a non-claiming view."""
    require(report.get("schema") == SCHEMA, "schema")
    require(report.get("status") == "sealed_model_or_saved_comparison", "sealed status")
    require(report.get("gate_modified") is False, "gate modification")
    require(report.get("physical_execution") is False
            and report.get("cuda_initialized") is False, "execution boundary")
    require(report.get("outcomes_opened") is False, "outcome boundary")
    require(report.get("P1_admission") is False, "P1 must remain false")
    require(report.get("mechanism_claim") == "not_established",
            "P1=false forbids mechanism claim")
    roots = report.get("roots")
    require(isinstance(roots, list) and roots and
            all(type(root) is str and root for root in roots)
            and len(set(roots)) == len(roots), "root roster")
    rows = report.get("rows")
    expected = [(root, arm) for root in roots for arm in ARMS]
    require(isinstance(rows, list) and len(rows) == len(expected), "complete denominator")
    require([(row.get("root"), row.get("arm")) for row in rows] == expected,
            "fixed arm/root order")
    for row, (root, arm) in zip(rows, expected):
        _row(row, root, arm)
    require(report.get("comparison_direction") == "full_composition_minus_control",
            "comparison direction")
    require(report.get("failure_policy") == {
        "retain_row": True,
        "retain_partial": True,
        "no_retry": True,
        "no_replacement": True,
        "no_imputation": True,
        "pending_is_not_complete": True,
        "failed_is_not_success": True,
    }, "failure policy")
    return {
        "schema": NAME + ".contract.v1",
        "roots": list(roots),
        "arms": list(ARMS),
        "resource_axes": list(RESOURCE_AXES),
        "rows": rows,
        "comparison_direction": report["comparison_direction"],
        "P1_admission": False,
        "P2_admission": False,
        "mechanism_claim": "not_established",
        "scientific_admission": False,
        "physical_execution": False,
        "outcomes_opened": False,
        "gate_modified": False,
        "failure_policy": report["failure_policy"],
    }


def build_from_bytes(raw, expected_sha):
    require(digest(raw) == expected_sha, "report SHA")
    return build_contract(decode(raw))
