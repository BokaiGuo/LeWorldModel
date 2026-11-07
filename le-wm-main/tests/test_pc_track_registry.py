import pytest

from experiments.pc_input_permissions import RGB, STATE, freeze_audit, manifest
from experiments.pc_track_registry import eligible, empty_gate_ledger


@pytest.fixture
def record(tmp_path):
    path = tmp_path / "fixture"
    path.write_text("synthetic binding")
    audit = freeze_audit(RGB, "lewm", [path], preprocessing="fixture", clock="fixture")
    receipts = [{"track_id": RGB, "input_audit_hash": audit.fingerprint,
                 "operation": "encode", "inputs": {"pixels": {}}}]
    value = manifest(audit, algorithm_version="test", protocol_path=path,
                     task_manifest_path=path, receipts=receipts)
    kwargs = dict(target_track=RGB, algorithm_version="test", protocol_hash=value["protocol_hash"],
                  task_manifest_hash=value["task_manifest_hash"])
    return value, kwargs


def test_eligible_is_not_gate_passage(record):
    value, kwargs = record
    assert eligible(value, **kwargs)
    assert all(g["scientific_status"] == "not_evaluated"
               for g in empty_gate_ledger(RGB, "test")["gates"].values())
    assert not empty_gate_ledger(STATE, "test")["supports_primary_rgb_claim"]


@pytest.mark.parametrize("field,value", [
    ("execution_status", "failed"), ("execution_status", "running"),
    ("scientific_status", "passed"), ("task_manifest_hash", "different"),
    ("track_id", STATE),
])
def test_reject_invalid_evidence(record, field, value):
    data, kwargs = record
    data[field] = value
    with pytest.raises(ValueError):
        eligible(data, **kwargs)


def test_reject_hidden_input_receipt(record):
    data, kwargs = record
    data["input_receipts"][0]["inputs"]["state"] = {}
    with pytest.raises(ValueError):
        eligible(data, **kwargs)
