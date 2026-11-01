import copy
import json

import pytest

from experiments.spectral_reachability import (
    p2_matched_composition_contract_20261102_exclusive as subject,
)


def budget():
    return {
        axis: {
            "planned": 10, "attempted": 0, "completed": 0,
            "failed": 0, "pending": 0, "unknown": 0,
        }
        for axis in subject.RESOURCE_AXES
    }


def report():
    rows = []
    for root in ("root0", "root1"):
        for arm in subject.ARMS:
            rows.append({
                "root": root, "arm": arm, "status": "completed",
                "budget": budget(),
                "failure": {
                    "kind": "none", "message": "", "retryable": False,
                    "rerun_authorized": False,
                },
                "outcome": None,
            })
    return {
        "schema": subject.SCHEMA,
        "status": "sealed_model_or_saved_comparison",
        "gate_modified": False,
        "physical_execution": False,
        "cuda_initialized": False,
        "outcomes_opened": False,
        "P1_admission": False,
        "mechanism_claim": "not_established",
        "roots": ["root0", "root1"],
        "rows": rows,
        "comparison_direction": "full_composition_minus_control",
        "failure_policy": {
            "retain_row": True, "retain_partial": True, "no_retry": True,
            "no_replacement": True, "no_imputation": True,
            "pending_is_not_complete": True, "failed_is_not_success": True,
        },
    }


def test_fixed_roster_budget_and_nonclaiming_contract():
    result = subject.build_contract(report())
    assert [row["arm"] for row in result["rows"][:4]] == list(subject.ARMS)
    assert result["P1_admission"] is False
    assert result["P2_admission"] is False
    assert result["mechanism_claim"] == "not_established"
    assert result["failure_policy"]["no_imputation"] is True


def test_failed_slot_is_retained_and_never_retried():
    value = report()
    row = value["rows"][1]
    row["status"] = "failed"
    row["failure"] = {
        "kind": "encoder_error", "message": "sealed failure",
        "retryable": True, "rerun_authorized": False,
    }
    row["budget"]["predict"].update(
        planned=10, attempted=1, completed=0, failed=1, pending=0, unknown=0
    )
    result = subject.build_contract(value)
    assert len(result["rows"]) == 8
    assert result["rows"][1]["status"] == "failed"


def test_pending_and_p1_claims_fail_closed():
    value = report()
    value["rows"][0]["status"] = "timeout"
    value["rows"][0]["failure"] = {
        "kind": "wall_timeout", "message": "pending callback",
        "retryable": True, "rerun_authorized": False,
    }
    value["rows"][0]["budget"]["predict"].update(
        planned=10, attempted=2, completed=1, failed=0, pending=1, unknown=1
    )
    assert subject.build_contract(value)["rows"][0]["status"] == "timeout"
    bad = copy.deepcopy(value)
    bad["P1_admission"] = True
    with pytest.raises(ValueError, match="P1"):
        subject.build_contract(bad)


def test_no_gate_change_or_outcome_opening():
    value = report()
    value["gate_modified"] = True
    with pytest.raises(ValueError, match="gate"):
        subject.build_contract(value)
    value = report()
    value["rows"][0]["outcome"] = {"success": True}
    with pytest.raises(ValueError, match="outcomes"):
        subject.build_contract(value)
