from __future__ import annotations

import json

import pytest

from experiments.audit_tworoom_action_effect_repairability import audit


def _report() -> dict:
    arms = {
        name: {
            "outcome": {"root_regret": {"3": 0.0}},
        }
        for name in ("frozen_visual", "effect_adapter", "action_only", "no_dynamics")
    }
    closed = {name: {"roots": [3]} for name in arms}
    return {
        "schema": "tworoom_action_effect_repairability.v1",
        "protocol": {
            "device": "cuda", "root_ids": {"train": [1], "calibration": [2], "outcome": [3]},
            "candidate_bank": {"count": 2, "same_world_fixed_bank_all_roots": True},
            "runtime_state_access": False, "training_outcome_roots_read": False,
            "outcome_used_for_selection": False,
            "outcome_execution_order": "frozen train/calibration adapter fit precedes first outcome counterfactual replay",
            "closed_loop": {"all_arms_same_policy_input_and_bank": True},
        },
        "ranking": arms, "closed_loop": closed, "claim_boundary": "test",
    }


def test_audit_accepts_complete_contract(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    assert audit(path)["status"] == "passed"


def test_audit_rejects_outcome_training_access(tmp_path) -> None:
    report = _report()
    report["protocol"]["training_outcome_roots_read"] = True
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="outcome-training"):
        audit(path)


def test_audit_requires_a_frozen_receipt_for_an_optional_support_gated_arm(tmp_path) -> None:
    report = _report()
    report["protocol"]["support_gated_repair"] = True
    report["ranking"]["support_gated"] = {"outcome": {"root_regret": {"3": 0.0}}}
    report["closed_loop"]["support_gated"] = {"roots": [3]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen gate evidence"):
        audit(path)
