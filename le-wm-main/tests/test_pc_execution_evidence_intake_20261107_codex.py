import json

import pytest
import torch

from experiments.pc_execution_evidence_intake_20261107_codex import intake
from experiments.pc_input_permissions import RGB, STATE, freeze_audit, manifest, tensor_receipt


@pytest.fixture
def example(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixture")
    audit = freeze_audit(RGB, "lewm", [source], preprocessing="fixture", clock="fixture")
    receipt = {"track_id": RGB, "input_audit_hash": audit.fingerprint, "operation": "score",
               "inputs": {key: tensor_receipt(torch.zeros(1))
                          for key in ("pixels", "goal", "action", "candidate_actions")}}
    record = manifest(audit, algorithm_version="fixture", protocol_path=source,
                      task_manifest_path=source, receipts=[receipt])
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    args = {"track_id": RGB, "algorithm_version": "fixture", "protocol_path": source,
            "task_manifest_path": source, "gate": "PC4"}
    return path, record, args


def test_intake_never_promotes_scientific_state(example):
    path, _, args = example
    result = intake(path, **args)
    assert result["scientific_status"] == "not_evaluated"
    assert result["required_operations"] == ["score"]
    assert result["evidence_scope"] == "visual_closed_loop"


@pytest.mark.parametrize("change", ["cross_track", "missing_score", "missing_goal", "bad_gate",
                                   "passed_science", "bad_hash"])
def test_invalid_evidence_rejected(example, change):
    path, record, args = example
    if change == "cross_track":
        args["track_id"] = STATE
    elif change == "missing_score":
        receipt = record["input_receipts"][0]
        receipt["operation"] = "encode"
        receipt["inputs"] = {"pixels": receipt["inputs"]["pixels"]}
    elif change == "missing_goal":
        del record["input_receipts"][0]["inputs"]["goal"]
    elif change == "bad_gate":
        args["gate"] = "PC9"
    elif change == "passed_science":
        record["scientific_status"] = "passed"
    else:
        record["input_receipts"][0]["inputs"]["pixels"]["sha256"] = "bad"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        intake(path, **args)
