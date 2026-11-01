"""Exclusive, read-only P2 matched-comparison protocol contract.

This module binds existing P2 asset identities to a four-arm comparison:
full composition, no-composition SRA, zero, and a strong baseline.  It does
not load result files, read outcomes, launch a model/simulator, or authorize
P1/P2/mechanism claims.
"""
from __future__ import annotations

from copy import deepcopy
import math


SCHEMA = "p2_matched_comparison_protocol_20261102_exclusive.v1"
ARMS = ("full_composition", "no_composition_sra", "zero", "strong_baseline")
EXISTING_ASSET_IDS = {
    "full_composition": "p2_matched_composition_20261027",
    "no_composition_sra": "p2_no_composition_sra_20261027_v2",
    "zero": "p2_full20_sampling_20261027_v2.zero_slot",
    "strong_baseline": "agent_p2_strong_baseline_saved_summary_20261028_v3",
}
BUDGET_FIELDS = (
    "planning_predict_calls",
    "planning_predict_samples",
    "replay_predict_calls",
    "replay_predict_samples",
    "candidate_evaluations",
    "action_encoder_calls",
    "jacobian_calls",
    "vjp_directions",
    "svd_calls",
    "bounded_solve_calls",
    "backward_calls",
    "optimizer_steps",
    "physical_env_steps",
    "wall_time_limit_s",
)
FAILURE_POLICY = {
    "charge_before_dispatch": True,
    "retain_failed_attempt": True,
    "retain_aborted_attempt": True,
    "refund_failed_work": False,
    "retry_failed_attempt": False,
    "replace_missing_with_zero": False,
    "select_winner_if_any_failure": False,
    "require_all_slots": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            f"{label} must be a lowercase SHA256")


def _budget(value):
    require(isinstance(value, dict) and set(value) == set(BUDGET_FIELDS),
            "complete fixed budget fields required")
    for field in BUDGET_FIELDS:
        number = value[field]
        if field == "wall_time_limit_s":
            require(type(number) in (int, float) and math.isfinite(number) and number > 0,
                    f"known positive {field} required")
        else:
            require(type(number) is int and number >= 0,
                    f"known nonnegative integer {field} required")
    require(value["candidate_evaluations"] > 0, "candidate budget must be nonempty")
    require(value["planning_predict_calls"] == value["planning_predict_samples"],
            "planning call/sample accounting mismatch")
    require(value["replay_predict_calls"] == value["replay_predict_samples"],
            "replay call/sample accounting mismatch")
    require(value["action_encoder_calls"] ==
            value["planning_predict_samples"] + value["replay_predict_samples"],
            "action encoder accounting mismatch")


def _asset(value, arm):
    require(isinstance(value, dict) and set(value) == {
        "asset_id", "schema", "source_sha256", "outcome_accessed", "P1_admission"
    }, f"exact existing P2 asset reference required for {arm}")
    require(value["asset_id"] == EXISTING_ASSET_IDS[arm],
            f"{arm} must bind the existing P2 asset")
    require(isinstance(value["schema"], str) and value["schema"], f"{arm} asset schema required")
    _sha(value["source_sha256"], f"{arm} asset source")
    require(value["outcome_accessed"] is False, f"{arm} outcome access must be false")
    require(value["P1_admission"] is False, f"{arm} P1 admission must be false")


def _slots(value, arm, candidate_count):
    require(isinstance(value, list) and len(value) == candidate_count,
            f"{arm} must retain every frozen candidate slot")
    identities = []
    for slot in value:
        require(isinstance(slot, dict) and set(slot) == {"slot_id", "root_id"},
                f"{arm} candidate slot fields required")
        for field in ("slot_id", "root_id"):
            require(isinstance(slot[field], str) and slot[field], f"{arm} slot identity required")
        identities.append((slot["root_id"], slot["slot_id"]))
    require(len(set(identities)) == len(identities), f"{arm} duplicate candidate slot")
    return set(identities)


def _validate(protocol):
    require(isinstance(protocol, dict), "protocol object required")
    require(set(protocol) == {
        "schema", "selection", "cohort", "arms", "failure_policy", "p1_gate", "new_work"
    }, "outcome-bearing or unknown protocol fields are forbidden")
    require(protocol["schema"] == SCHEMA, "unsupported P2 protocol schema")
    require(protocol["selection"] == {
        "rule": "pre_outcome_frozen",
        "outcome_accessed": False,
        "selected_after_outcome": False,
    }, "selection must be frozen before outcome access")

    cohort = protocol["cohort"]
    require(isinstance(cohort, dict) and set(cohort) == {
        "cohort_id", "root_ids", "goal_ids", "context_ids"
    }, "complete matched cohort required")
    for field in ("cohort_id",):
        require(isinstance(cohort[field], str) and cohort[field], f"{field} required")
    ids = {}
    for field in ("root_ids", "goal_ids", "context_ids"):
        value = cohort[field]
        require(isinstance(value, list) and value and
                all(isinstance(item, str) and item for item in value),
                f"{field} required")
        require(len(value) == len(set(value)), f"duplicate {field}")
        ids[field] = set(value)
    require(len(ids["root_ids"]) == len(ids["goal_ids"]) == len(ids["context_ids"]),
            "matched cohort cardinality mismatch")

    require(isinstance(protocol["arms"], dict) and set(protocol["arms"]) == set(ARMS),
            "exact four-arm P2 comparison required")
    common_slots = None
    budgets = {}
    for arm in ARMS:
        spec = protocol["arms"][arm]
        require(isinstance(spec, dict) and set(spec) == {"asset", "budget", "candidate_slots"},
                f"exact arm fields required for {arm}")
        _asset(spec["asset"], arm)
        _budget(spec["budget"])
        slots = _slots(spec["candidate_slots"], arm, spec["budget"]["candidate_evaluations"])
        if common_slots is None:
            common_slots = slots
        require(slots == common_slots, f"matched root/slot census mismatch for {arm}")
        budgets[arm] = spec["budget"]

    require(protocol["failure_policy"] == FAILURE_POLICY, "failure policy is not fixed")
    gate = protocol["p1_gate"]
    require(isinstance(gate, dict) and set(gate) == {
        "P1_admission", "mechanism_claim_allowed", "reason"
    }, "complete P1 gate required")
    require(gate["P1_admission"] is False and gate["mechanism_claim_allowed"] is False
            and gate["reason"] == "P1_not_passed", "P1 must block mechanism claims")

    work = protocol["new_work"]
    require(work == {"neural_calls": 0, "physical_steps": 0, "outcomes_read": 0},
            "this contract cannot authorize new work or outcome reads")
    return budgets


def validate_protocol(protocol):
    """Validate structure and return a non-admitting comparison decision."""
    data = deepcopy(protocol)
    budgets = _validate(data)
    unique_budgets = {repr(sorted(value.items())) for value in budgets.values()}
    reasons = []
    if len(unique_budgets) != 1:
        reasons.append("budget_mismatch")
    return {
        "schema": SCHEMA + ".decision",
        "status": "blocked_p1",
        "arms": list(ARMS),
        "matched_root_slot_pairs": len(data["arms"][ARMS[0]]["candidate_slots"]),
        "budget_comparison_admissible": not reasons,
        "blocking_reasons": reasons + ["P1_not_passed"],
        "mechanism_claim_allowed": False,
        "P1_admission": False,
        "P2_admission": False,
        "physical_efficacy_verified": False,
        "outcome_accessed": False,
        "new_work": deepcopy(data["new_work"]),
        "failure_policy": deepcopy(data["failure_policy"]),
    }
