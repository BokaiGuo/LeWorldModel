"""Exclusive P2 matched-comparison contract tests.

These are structural tests only: no model, simulator, physical outcome, or
new result file is read.
"""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import (
    p2_matched_comparison_contract_20261102_exclusive as m,
)


def protocol():
    budget = {
        "planning_predict_calls": 80,
        "planning_predict_samples": 80,
        "replay_predict_calls": 4,
        "replay_predict_samples": 4,
        "candidate_evaluations": 40,
        "action_encoder_calls": 84,
        "jacobian_calls": 0,
        "vjp_directions": 0,
        "svd_calls": 0,
        "bounded_solve_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "physical_env_steps": 0,
        "wall_time_limit_s": 600.0,
    }
    arms = {}
    for arm in m.ARMS:
        arms[arm] = {
            "asset": {
                "asset_id": m.EXISTING_ASSET_IDS[arm],
                "schema": f"existing.{arm}.report",
                "source_sha256": "a" * 64,
                "outcome_accessed": False,
                "P1_admission": False,
            },
            "budget": deepcopy(budget),
            "candidate_slots": [
                {"slot_id": f"root-0-slot-{i}", "root_id": "root-0"}
                for i in range(40)
            ],
        }
    return {
        "schema": m.SCHEMA,
        "selection": {
            "rule": "pre_outcome_frozen",
            "outcome_accessed": False,
            "selected_after_outcome": False,
        },
        "cohort": {
            "cohort_id": "p2-independent-cohort-v1",
            "root_ids": ["root-0"],
            "goal_ids": ["goal-0"],
            "context_ids": ["context-0"],
        },
        "arms": arms,
        "failure_policy": {
            "charge_before_dispatch": True,
            "retain_failed_attempt": True,
            "retain_aborted_attempt": True,
            "refund_failed_work": False,
            "retry_failed_attempt": False,
            "replace_missing_with_zero": False,
            "select_winner_if_any_failure": False,
            "require_all_slots": True,
        },
        "p1_gate": {
            "P1_admission": False,
            "mechanism_claim_allowed": False,
            "reason": "P1_not_passed",
        },
        "new_work": {
            "neural_calls": 0,
            "physical_steps": 0,
            "outcomes_read": 0,
        },
    }


def test_valid_protocol_is_blocked_at_p1_and_has_no_execution():
    result = m.validate_protocol(protocol())
    assert result["status"] == "blocked_p1"
    assert result["P1_admission"] is False
    assert result["P2_admission"] is False
    assert result["mechanism_claim_allowed"] is False
    assert result["new_work"] == {"neural_calls": 0, "physical_steps": 0, "outcomes_read": 0}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["selection"].update(outcome_accessed=True),
        lambda p: p["selection"].update(selected_after_outcome=True),
        lambda p: p["new_work"].update(outcomes_read=1),
        lambda p: p["failure_policy"].update(refund_failed_work=True),
        lambda p: p["failure_policy"].update(replace_missing_with_zero=True),
        lambda p: p["failure_policy"].update(select_winner_if_any_failure=True),
        lambda p: p["arms"]["no_composition_sra"]["candidate_slots"].pop(),
        lambda p: p["arms"]["zero"]["asset"].update(outcome_accessed=True),
        lambda p: p["p1_gate"].update(mechanism_claim_allowed=True),
    ],
)
def test_contract_fails_closed_on_outcome_leak_budget_or_failure_policy(mutate):
    data = protocol()
    mutate(data)
    with pytest.raises(ValueError):
        m.validate_protocol(data)


def test_unknown_or_missing_budget_is_not_zero():
    data = protocol()
    data["arms"]["strong_baseline"]["budget"]["svd_calls"] = None
    with pytest.raises(ValueError, match="known"):
        m.validate_protocol(data)


def test_arm_budget_vectors_are_not_silently_equalized():
    data = protocol()
    data["arms"]["no_composition_sra"]["budget"]["svd_calls"] = 2
    result = m.validate_protocol(data)
    assert result["budget_comparison_admissible"] is False
    assert "budget_mismatch" in result["blocking_reasons"]


def test_protocol_does_not_accept_physical_outcomes_or_claim_mechanism():
    data = protocol()
    data["physical_observations"] = []
    with pytest.raises(ValueError, match="outcome"):
        m.validate_protocol(data)
    data = protocol()
    data["p1_gate"]["P1_admission"] = True
    with pytest.raises(ValueError):
        m.validate_protocol(data)


def test_identity_join_is_by_root_and_slot_not_array_position():
    data = protocol()
    data["arms"]["zero"]["candidate_slots"].reverse()
    assert m.validate_protocol(data)["status"] == "blocked_p1"
    data["arms"]["zero"]["candidate_slots"][0]["slot_id"] = "root-0-slot-999"
    with pytest.raises(ValueError, match="slot"):
        m.validate_protocol(data)
