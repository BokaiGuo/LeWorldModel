"""Negative audit: accepted malformed fixtures are defects, not validation.

These tests intentionally pass when the existing draft accepts the named
counterexample. They document current behavior; they are not acceptance tests
and do not establish physical evidence, source provenance, or stage admission.
"""
from copy import deepcopy

import pytest

from experiments.spectral_reachability.d2p_p1_p2_p3_interface_contract_20261102_codexexclusive import (
    validate,
)
from test_d2p_p1_p2_p3_interface_contract_20261102_codexexclusive import p1, p2, p3


def test_p1_evidence_booleans_are_not_identity_bound():
    receipt = p1()
    receipt["evidence"] = {
        "non_greedy_prefix": True,
        "own_state_capability": True,
        "closed_loop_outcome": True,
    }
    receipt["outcome_rows"][0]["status"] = "failed"
    assert validate("p1", receipt)["outcome_rows"][0]["status"] == "failed"


def test_p2_accepts_disconnected_edge_contexts():
    receipt = p2()
    second = deepcopy(receipt["edges"][0])
    second["edge_id"] = "e1"
    second["input_context"] = {"rgb": "unrelated"}
    receipt["edges"].append(second)
    result = validate("p2", receipt)
    assert result["edges"][0]["observed_context"] != result["edges"][1]["input_context"]


@pytest.mark.parametrize(
    "field",
    ["input_context", "predicted_postcondition", "observed_context",
     "action_witness", "replay_receipt"],
)
def test_p2_accepts_null_mandatory_edge_fields(field):
    receipt = p2()
    receipt["edges"][0][field] = None
    assert validate("p2", receipt)["edges"][0][field] is None


def test_p2_failed_edge_can_self_declare_recompile():
    receipt = p2()
    receipt["edges"][0]["status"] = "failed"
    assert validate("p2", receipt)["edges"][0]["status"] == "failed"


def test_p3_accepts_negative_fractional_schedule_coordinates():
    receipt = p3()
    receipt["signal"].update(horizon=-1.5, depth=-2.0, predict_budget=-3.0)
    result = validate("p3", receipt)
    assert result["signal"]["horizon"] < 0
    assert result["signal"]["predict_budget"] != result["allocation"]["planned"]


def test_rejected_status_is_returned_as_successful_validation():
    receipt = p1()
    receipt["status"] = "contract_rejected"
    assert validate("p1", receipt)["status"] == "contract_rejected"
