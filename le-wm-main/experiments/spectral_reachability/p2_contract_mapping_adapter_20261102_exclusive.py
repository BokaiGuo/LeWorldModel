"""Outcome-free field mapping between the two independent P2 contracts.

Pure in-memory translation only. The supplied protocols and rows are
unverified assertions; this module establishes no asset, identity, budget,
execution, or scientific authority.
"""
from __future__ import annotations

from copy import deepcopy

from . import p2_matched_comparison_contract_20261102_exclusive as protocol
from . import p2_matched_composition_contract_20261102_exclusive as ledger

SCHEMA = "p2_contract_mapping_adapter_20261102_exclusive.v1"

# These correspond by field name and intended unit only. Scope and metering
# remain unverified: the protocol is per arm and the ledger is per root/arm.
DIRECT_FIELDS = {
    "action_encoder": "action_encoder_calls",
    "jacobian": "jacobian_calls",
    "vjp": "vjp_directions",
    "svd": "svd_calls",
    "bounded_solve": "bounded_solve_calls",
    "backward": "backward_calls",
    "wall_seconds": "wall_time_limit_s",
}
UNMAPPED_PROTOCOL = (
    "planning_predict_samples", "replay_predict_samples",
    "candidate_evaluations", "optimizer_steps", "physical_env_steps",
)
UNMAPPED_LEDGER = ("forward_ad",)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def map_contracts(protocol_input, ledger_input):
    """Map validated *structures*, with no inference that either is authentic.

    No data are loaded or dispatched. In particular the caller must not use
    this projection as a source or seal verification, an execution permit, or a
    computation-matching verdict.
    """
    p = deepcopy(protocol_input)
    r = deepcopy(ledger_input)
    # Explicit input surface: avoid accepting and returning outcome-bearing
    # fields that the older per-row validator ignores.
    _require(isinstance(r, dict) and set(r) == {
        "schema", "status", "gate_modified", "physical_execution",
        "cuda_initialized", "outcomes_opened", "P1_admission",
        "mechanism_claim", "roots", "rows", "comparison_direction",
        "failure_policy",
    }, "unexpected ledger fields")
    _require(isinstance(r["rows"], list) and all(
        isinstance(row, dict) and set(row) == {
            "root", "arm", "status", "budget", "failure", "outcome"
        } for row in r["rows"]
    ), "unexpected row fields")

    protocol.validate_protocol(p)
    ledger.build_contract(r)
    _require(protocol.ARMS == ledger.ARMS, "contract arm drift")
    _require(set(p["cohort"]["root_ids"]) == set(r["roots"]), "root roster mismatch")
    root_set = set(r["roots"])
    for arm in protocol.ARMS:
        _require(all(slot["root_id"] in root_set
                     for slot in p["arms"][arm]["candidate_slots"]),
                 "candidate slot outside root roster")

    rows = []
    for row in r["rows"]:
        arm = row["arm"]
        b = p["arms"][arm]["budget"]
        comparisons = {
            axis: {
                "protocol_field": field,
                "protocol_arm_value": b[field],
                "ledger_root_arm_planned": row["budget"][axis]["planned"],
                "scope_equivalence_verified": False,
            }
            for axis, field in DIRECT_FIELDS.items()
        }
        comparisons["predict"] = {
            "protocol_fields": ["planning_predict_calls", "replay_predict_calls"],
            "protocol_arm_values": [b["planning_predict_calls"], b["replay_predict_calls"]],
            "ledger_root_arm_planned": row["budget"]["predict"]["planned"],
            "scope_equivalence_verified": False,
        }
        rows.append({
            "root": row["root"], "arm": arm,
            "candidate_slots_for_root": deepcopy([
                slot for slot in p["arms"][arm]["candidate_slots"]
                if slot["root_id"] == row["root"]
            ]),
            "asset_reference_unverified": deepcopy(p["arms"][arm]["asset"]),
            "protocol_arm_budget": deepcopy(b),
            "ledger_root_arm_budget": deepcopy(row["budget"]),
            "field_correspondences": comparisons,
            "status": row["status"],
            "failure": deepcopy(row["failure"]),
        })
    return {
        "schema": SCHEMA,
        "input_schemas": {"protocol": p["schema"], "ledger": r["schema"]},
        "cohort_reference_unverified": deepcopy(p["cohort"]),
        "root_roster": list(r["roots"]),
        "arm_roster": list(protocol.ARMS),
        "rows": rows,
        "unmapped_protocol_fields": list(UNMAPPED_PROTOCOL),
        "unmapped_ledger_axes": list(UNMAPPED_LEDGER),
        "failure_policies": {
            "protocol": deepcopy(p["failure_policy"]),
            "ledger": deepcopy(r["failure_policy"]),
        },
        "structural_validation_only": True,
        "asset_verified": False,
        "seal_verified": False,
        "root_goal_context_identity_verified": False,
        "budget_scope_equivalence_verified": False,
        "budget_matched": None,
        "physical_execution_authorized": False,
        "P1_admission": False,
        "P2_admission": False,
        "mechanism_claim_allowed": False,
    }
