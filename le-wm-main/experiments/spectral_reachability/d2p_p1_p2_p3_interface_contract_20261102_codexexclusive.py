"""Exclusive fail-closed interface contracts for Planning Compiler P1/P2/P3.

This module validates receipt shape and ordering only.  It does not run a
model, simulator, environment, GPU job, or historical result.  Passing these
validators means that a future producer emitted an auditable receipt; it does
not establish physical efficacy or stage admission.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite


SCHEMAS = {
    "p1": "d2p_p1_admission_contract.v1",
    "p2": "d2p_p2_composition_search_contract.v1",
    "p3": "d2p_p3_dynamic_schedule_contract.v1",
}

ADMISSION_FIELDS = ("P1_admission", "P2_admission", "P3_admission")
SHA256_FIELDS = (
    "checkpoint_sha256",
    "task_manifest_sha256",
    "protocol_sha256",
    "source_sha256",
)


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _sha(value, name):
    _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be finite numeric")


def _identity(payload):
    identity = _mapping(payload.get("identity"), "identity")
    for field in ("root_id", "task_id", "protocol_id", "frozen_model_id"):
        _text(identity.get(field), f"identity.{field}")
    for field in SHA256_FIELDS:
        _sha(identity.get(field), f"identity.{field}")
    if identity["root_id"] == identity["source_sha256"]:
        raise ValueError("root identity must not be replaced by source identity")
    return identity


def _admission_false(payload):
    for field in ADMISSION_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"{field} must remain false in an interface receipt")


def _status(payload):
    status = payload.get("status")
    if status not in ("contract_validated", "contract_rejected"):
        raise ValueError("status must be an explicit contract status")


def _root_set(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} must be a nonempty list")
    roots = set()
    for row in rows:
        item = _mapping(row, name)
        _text(item.get("root_id"), f"{name}.root_id")
        if item["root_id"] in roots:
            raise ValueError(f"{name} contains duplicate root identity")
        roots.add(item["root_id"])
    return roots


def validate_p1(payload):
    """Validate the pre-outcome P1 admission receipt shape.

    The contract requires one identity-bound row to join non-greedy evidence,
    own-state capability evidence, and an independent suffix/closed-loop
    outcome.  It deliberately rejects a historical positive gate.
    """
    payload = deepcopy(_mapping(payload, "P1 receipt"))
    if payload.get("schema") != SCHEMAS["p1"]:
        raise ValueError("wrong P1 contract schema")
    _status(payload)
    _admission_false(payload)
    identity = _identity(payload)
    if payload.get("outcome_sealed_before_selection") is not True:
        raise ValueError("P1 outcome must be sealed before candidate selection")
    if payload.get("source_disjoint_outcome") is not True:
        raise ValueError("P1 requires an explicit source-disjoint outcome")
    rows = _mapping(payload.get("evidence"), "evidence")
    required = ("non_greedy_prefix", "own_state_capability", "closed_loop_outcome")
    if any(rows.get(field) is not True for field in required):
        raise ValueError("P1 requires all three same-root evidence components")
    roots = _root_set(payload.get("outcome_rows"), "outcome_rows")
    if identity["root_id"] not in roots:
        raise ValueError("P1 identity root is absent from outcome rows")
    controls = payload.get("strong_baseline_controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("P1 strong-baseline matrix is required")
    if any(not isinstance(control, str) or not control.strip() for control in controls):
        raise ValueError("P1 baseline names must be nonempty strings")
    return payload


def validate_p2(payload):
    """Validate a searched multi-program composition receipt.

    This is stricter than replaying already supplied edges: it requires search
    and a predeclared no-composition/flat-search comparator, while retaining
    failures and partial executions in the denominator.
    """
    payload = deepcopy(_mapping(payload, "P2 receipt"))
    if payload.get("schema") != SCHEMAS["p2"]:
        raise ValueError("wrong P2 contract schema")
    _status(payload)
    _admission_false(payload)
    _identity(payload)
    if payload.get("candidate_library_sealed_before_outcome") is not True:
        raise ValueError("P2 candidate library must be sealed before outcome")
    if payload.get("search_executed") is not True:
        raise ValueError("P2 receipt must represent search, not verification only")
    if payload.get("same_budget_and_feedback") is not True:
        raise ValueError("P2 comparator must share budget and feedback")
    edges = payload.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("P2 requires at least one searched edge")
    edge_ids = set()
    for edge in edges:
        edge = _mapping(edge, "edges item")
        _text(edge.get("edge_id"), "edge_id")
        if edge["edge_id"] in edge_ids:
            raise ValueError("P2 edge IDs must be unique")
        edge_ids.add(edge["edge_id"])
        for field in (
            "input_context",
            "predicted_postcondition",
            "observed_context",
            "action_witness",
            "replay_receipt",
        ):
            if field not in edge:
                raise ValueError(f"P2 edge missing {field}")
        if edge["observed_context"] == edge["predicted_postcondition"] and edge.get(
            "observation_required"
        ) is not True:
            raise ValueError("P2 must record an actual observation boundary")
        if edge.get("recompiled_after_observation") is not True:
            raise ValueError("P2 must recompile after observation")
        if edge.get("status") not in ("completed", "failed", "partial", "recompiled"):
            raise ValueError("P2 edge status must retain failure/partial state")
    comparator = _mapping(payload.get("comparator"), "comparator")
    for field in ("no_composition", "flat_search"):
        if comparator.get(field) is not True:
            raise ValueError("P2 no-composition and flat-search controls are required")
    if payload.get("failed_or_partial_retained") is not True:
        raise ValueError("P2 cannot censor failed or partial branches")
    return payload


def validate_p3(payload):
    """Validate a dynamic schedule receipt without making a quality claim."""
    payload = deepcopy(_mapping(payload, "P3 receipt"))
    if payload.get("schema") != SCHEMAS["p3"]:
        raise ValueError("wrong P3 contract schema")
    _status(payload)
    _admission_false(payload)
    _identity(payload)
    if payload.get("schedule_sealed_before_outcome") is not True:
        raise ValueError("P3 schedule rule must be frozen before outcome")
    if payload.get("same_feedback_and_budget_cap") is not True:
        raise ValueError("P3 requires common feedback and budget cap")
    if payload.get("allocation_before_outcome") is not True:
        raise ValueError("P3 allocation must precede outcome")
    signal = _mapping(payload.get("signal"), "signal")
    if signal.get("source") not in ("model_dynamics", "declared_geometry"):
        raise ValueError("P3 signal source must be declared dynamics/geometry")
    if signal.get("read_after_outcome") is not False:
        raise ValueError("P3 signal must not read outcome")
    for field in ("horizon", "depth", "predict_budget"):
        if field not in signal:
            raise ValueError(f"P3 signal missing {field}")
        _finite_number(signal[field], f"signal.{field}")
    allocation = _mapping(payload.get("allocation"), "allocation")
    for field in ("planned", "actual", "unused_budget"):
        _nonnegative_int(allocation.get(field), f"allocation.{field}")
    if allocation["actual"] > allocation["planned"]:
        raise ValueError("actual allocation cannot exceed planned allocation")
    if allocation["unused_budget"] != allocation["planned"] - allocation["actual"]:
        raise ValueError("unused budget must be explicit arithmetic remainder")
    if payload.get("failed_work_refunded") is not False:
        raise ValueError("failed work must not be refunded")
    if payload.get("unknown_work_is_zero") is not False:
        raise ValueError("unknown work cannot be encoded as zero")
    controls = _mapping(payload.get("controls"), "controls")
    for field in ("fixed_horizon", "fixed_budget", "randomized_schedule"):
        if controls.get(field) is not True:
            raise ValueError("P3 fixed/randomized comparison controls are required")
    return payload


def validate(kind, payload):
    """Dispatch a contract validator and return an owned validated snapshot."""
    try:
        validator = {"p1": validate_p1, "p2": validate_p2, "p3": validate_p3}[kind]
    except KeyError as exc:
        raise ValueError("kind must be one of p1, p2, p3") from exc
    return validator(payload)
