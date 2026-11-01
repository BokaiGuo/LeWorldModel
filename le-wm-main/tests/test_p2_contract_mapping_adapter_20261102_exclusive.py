"""Synthetic, in-memory mapping tests; no models or outcome files."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import (
    p2_contract_mapping_adapter_20261102_exclusive as m,
)


def inputs():
    roots = ["r0", "r1"]
    p_budget = dict.fromkeys(m.protocol.BUDGET_FIELDS, 0)
    p_budget.update(planning_predict_calls=8, planning_predict_samples=8,
                    replay_predict_calls=2, replay_predict_samples=2,
                    action_encoder_calls=10, candidate_evaluations=2,
                    wall_time_limit_s=100)
    arms = {
        arm: {
            "asset": {"asset_id": m.protocol.EXISTING_ASSET_IDS[arm],
                      "schema": "example", "source_sha256": "a" * 64,
                      "outcome_accessed": False, "P1_admission": False},
            "budget": deepcopy(p_budget),
            "candidate_slots": [{"root_id": root, "slot_id": "s0"} for root in roots],
        }
        for arm in m.protocol.ARMS
    }
    p = {
        "schema": m.protocol.SCHEMA,
        "selection": {"rule": "pre_outcome_frozen", "outcome_accessed": False,
                      "selected_after_outcome": False},
        "cohort": {"cohort_id": "c", "root_ids": roots[:],
                   "goal_ids": ["g0", "g1"], "context_ids": ["c0", "c1"]},
        "arms": arms, "failure_policy": deepcopy(m.protocol.FAILURE_POLICY),
        "p1_gate": {"P1_admission": False, "mechanism_claim_allowed": False,
                    "reason": "P1_not_passed"},
        "new_work": {"neural_calls": 0, "physical_steps": 0, "outcomes_read": 0},
    }
    rows = []
    for root in roots:
        for arm in m.ledger.ARMS:
            b = {axis: {"planned": 10, "attempted": 0, "completed": 0,
                        "failed": 0, "pending": 0, "unknown": 0}
                 for axis in m.ledger.RESOURCE_AXES}
            rows.append({"root": root, "arm": arm, "status": "completed",
                         "budget": b, "failure": {
                             "kind": "none", "message": "", "retryable": False,
                             "rerun_authorized": False}, "outcome": None})
    r = {
        "schema": m.ledger.SCHEMA, "status": "sealed_model_or_saved_comparison",
        "gate_modified": False, "physical_execution": False, "cuda_initialized": False,
        "outcomes_opened": False, "P1_admission": False,
        "mechanism_claim": "not_established", "roots": roots[:], "rows": rows,
        "comparison_direction": "full_composition_minus_control",
        "failure_policy": {"retain_row": True, "retain_partial": True, "no_retry": True,
                           "no_replacement": True, "no_imputation": True,
                           "pending_is_not_complete": True, "failed_is_not_success": True},
    }
    return p, r


def test_projection_preserves_scope_and_does_not_certify():
    p, r = inputs()
    result = m.map_contracts(p, r)
    assert len(result["rows"]) == 8
    row = result["rows"][0]
    assert row["field_correspondences"]["predict"]["protocol_arm_values"] == [8, 2]
    assert row["field_correspondences"]["predict"]["ledger_root_arm_planned"] == 10
    assert result["budget_matched"] is None
    assert not result["asset_verified"] and not result["seal_verified"]
    assert not result["root_goal_context_identity_verified"]
    assert not result["mechanism_claim_allowed"]
    assert "forward_ad" in result["unmapped_ledger_axes"]
    assert "planning_predict_samples" in result["unmapped_protocol_fields"]
    result["rows"][0]["ledger_root_arm_budget"]["predict"]["planned"] = 99
    assert r["rows"][0]["budget"]["predict"]["planned"] == 10


def test_failure_and_pending_are_preserved_not_promoted():
    p, r = inputs()
    row = r["rows"][1]
    row["status"] = "timeout"
    row["failure"].update(kind="timeout", message="pending", retryable=True)
    row["budget"]["predict"].update(attempted=2, completed=1, pending=1, unknown=1)
    actual = m.map_contracts(p, r)["rows"][1]
    assert actual["status"] == "timeout"
    assert actual["ledger_root_arm_budget"]["predict"]["pending"] == 1
    assert actual["failure"]["rerun_authorized"] is False


def test_roster_and_rogue_slot_rejected():
    p, r = inputs()
    r["roots"][1] = "other"
    with pytest.raises(ValueError):
        m.map_contracts(p, r)
    p, r = inputs()
    for arm in m.protocol.ARMS:
        p["arms"][arm]["candidate_slots"][1]["root_id"] = "other"
    with pytest.raises(ValueError, match="outside"):
        m.map_contracts(p, r)


@pytest.mark.parametrize("mutation", [
    lambda p, r: p["p1_gate"].update(mechanism_claim_allowed=True),
    lambda p, r: r.update(outcomes_opened=True),
    lambda p, r: r["rows"][0].update(outcome={"success": True}),
    lambda p, r: r["rows"][0].update(extra_outcome=1),
    lambda p, r: p["arms"]["zero"]["budget"].update(svd_calls=None),
])
def test_rejects_claims_outcomes_and_unknown_protocol_budget(mutation):
    p, r = inputs()
    mutation(p, r)
    with pytest.raises(ValueError):
        m.map_contracts(p, r)


def test_budget_difference_is_visible_not_equalized():
    p, r = inputs()
    p["arms"]["zero"]["budget"]["svd_calls"] = 3
    mapped = m.map_contracts(p, r)
    zero = mapped["rows"][2]["field_correspondences"]["svd"]
    assert zero["protocol_arm_value"] == 3
    assert zero["ledger_root_arm_planned"] == 10
    assert mapped["budget_matched"] is None
