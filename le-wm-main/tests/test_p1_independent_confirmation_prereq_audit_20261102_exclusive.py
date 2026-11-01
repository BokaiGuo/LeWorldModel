import copy
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import (
    p1_independent_confirmation_prereq_audit_20261102_exclusive as audit,
)


H = "a" * 64


def manifest():
    return {
        "schema": audit.SCHEMA,
        "candidate": {
            "candidate_id": "fresh-root-01",
            "role": "independent_confirmation",
            "source_namespace": "fresh-confirmation",
        },
        "root_identity": {
            "status": "verified",
            "root_id": "fresh-root-01",
            "reset_seed": 9001,
            "root_sha256": H,
            "goal_sha256": H,
            "action_history_sha256": H,
            "authority": "independent_authority",
            "historical_scope": "all_known_runs_and_restart_ledgers",
        },
        "source_receipt": {
            "status": "verified",
            "receipt_sha256": H,
            "queue_sha256": H,
            "reset_event_sha256": H,
            "producer_run_id": "fresh-run-01",
            "complete": True,
            "files": [{"path": "source.bin", "sha256": H, "role": "root_source"}],
        },
        "historical_dedup": {
            "status": "verified",
            "index_sha256": H,
            "scope": "all_known_runs_and_restart_ledgers",
            "root_sha256_checked": H,
            "goal_sha256_checked": H,
            "action_history_sha256_checked": H,
            "exact_duplicate_count": 0,
            "approximate_duplicate_check": True,
            "complete": True,
        },
        "checkpoint_provenance": {
            "status": "verified",
            "checkpoint_sha256": H,
            "config_sha256": H,
            "training_run_id": "train-01",
            "training_dataset_sha256": H,
            "training_split_manifest_sha256": H,
            "normalizer_sha256": H,
            "training_code_revision_sha256": H,
            "split_membership": "training_disjoint",
            "authoritative": True,
        },
        "negative_controls": {
            "dev303_used": False,
            "gate_modified": False,
            "outcome_used_for_selection": False,
        },
    }


def test_current_known_boundary_is_fail_closed():
    data = manifest()
    data["checkpoint_provenance"]["status"] = "unknown"
    data["checkpoint_provenance"]["authoritative"] = False
    data["historical_dedup"]["complete"] = False
    data["historical_dedup"]["status"] = "unknown"
    result = audit.audit(data)
    assert result["verdict"] == "FAIL_CLOSED"
    assert "checkpoint_training_split_provenance_unverified" in result["gaps"]
    assert "historical_dedup_incomplete_or_unverified" in result["gaps"]
    assert result["P1_independent_confirmation_authorized"] is False


def test_dev303_can_never_be_promoted():
    data = manifest()
    data["candidate"]["candidate_id"] = "dev303"
    data["candidate"]["source_namespace"] = "dev303"
    data["negative_controls"]["dev303_used"] = True
    result = audit.audit(data)
    assert result["verdict"] == "FAIL_CLOSED"
    assert "candidate_is_dev303_and_cannot_be_independent_confirmation" in result["gaps"]
    assert "negative_control_dev303_result_used" in result["gaps"]


def test_unknown_root_receipt_and_approximate_dedup_block():
    data = manifest()
    data["root_identity"]["status"] = "unknown"
    data["source_receipt"]["complete"] = False
    data["historical_dedup"]["approximate_duplicate_check"] = False
    result = audit.audit(data)
    assert result["verdict"] == "FAIL_CLOSED"
    assert len(result["gaps"]) == 3


def test_gate_control_is_a_negative_control_not_a_mutation():
    data = manifest()
    data["negative_controls"]["gate_modified"] = True
    result = audit.audit(data)
    assert "negative_control_gate_modified" in result["gaps"]
    assert result["operations"]["gate_mutation"] is False


def test_complete_contract_passes_only_with_all_authoritative_fields():
    result = audit.audit(manifest())
    assert result["verdict"] == "PASS"
    assert result["P1_independent_confirmation_authorized"] is True
    assert all(value is False for value in result["operations"].values())


def test_unknown_fields_and_duplicate_keys_are_rejected():
    data = manifest()
    data["root_identity"]["extra"] = 1
    with pytest.raises(ValueError):
        audit.audit(data)
    raw = '{"schema":"x","schema":"y"}'
    run = subprocess.run(
        [sys.executable, "-B", audit.__file__],
        input=raw,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "duplicate JSON key" in run.stdout


def test_cli_is_fail_closed_and_does_not_touch_runtime():
    data = manifest()
    data["checkpoint_provenance"]["split_membership"] = "unknown"
    run = subprocess.run(
        [sys.executable, "-B", audit.__file__],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        check=False,
    )
    output = json.loads(run.stdout)
    assert run.returncode == 2
    assert output["verdict"] == "FAIL_CLOSED"
    assert "checkpoint_training_split_provenance_unverified" not in output["gaps"]
    assert "checkpoint_training_split_membership_not_proven_disjoint" in output["gaps"]


def test_manifest_mutation_does_not_change_declared_negative_boundary():
    data = copy.deepcopy(manifest())
    data["negative_controls"]["outcome_used_for_selection"] = True
    result = audit.audit(data)
    assert result["verdict"] == "FAIL_CLOSED"
    assert "negative_control_outcome_used_for_selection" in result["gaps"]
