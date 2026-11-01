import hashlib
import json

import pytest

from experiments.spectral_reachability import codex_p1_h10_independent_join_20261030_v1 as m


def test_missing_and_tampered_source_fail_closed(tmp_path):
    path = tmp_path/"data.json"
    with pytest.raises(ValueError, match="missing"):
        m.read(path, "0"*64)
    path.write_text('{"value":1}', encoding="utf8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert m.read(path, sha) == {"value": 1}
    path.write_text('{"value":2}', encoding="utf8")
    with pytest.raises(ValueError, match="changed"):
        m.read(path, sha)


def test_cp5_binds_state_snapshot_and_original_file(tmp_path):
    path = tmp_path/"cp5.png"
    path.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    cp = {"step": 5, "state": [1], "snapshot": {"x": 1}, "file_sha256": digest,
          "pixel_sha256": "a"*64, "png": "cp5.png"}
    anchor = {"checkpoint": cp}
    row = {"start_verified": True, "start_state": [1], "checkpoints": [dict(cp)]}
    m.cp5(row, anchor, tmp_path)
    row["checkpoints"][0]["snapshot"] = {"x": 2}
    with pytest.raises(ValueError, match="snapshot"):
        m.cp5(row, anchor, tmp_path)
    row["checkpoints"][0]["snapshot"] = {"x": 1}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="actual frame"):
        m.cp5(row, anchor, tmp_path)


def test_per_branch_resource_identity_and_missing_are_rejected():
    anchor = {"index": 0, "condition": "off", "arm": "a", "role": "zero"}
    row = {"index": 0, "probe": "zero", "warmup_returned": 0, "own_returned": 1}
    def event(resource):
        return {"branch": 0, "anchor_index": 0, "condition": "off", "arm": "a",
                "candidate_role": "zero", "probe": "zero", "resource": resource, "status": "returned"}
    events = [event("constructor"), event("public_step"), event("close")]
    assert m.branch_counts(events, row, anchor)["public_step"] == {"attempted": 1, "completed": 1}
    with pytest.raises(ValueError, match="cost"):
        m.branch_counts(events[:-1], row, anchor)
    events[1]["candidate_role"] = "selected"
    with pytest.raises(ValueError, match="identity"):
        m.branch_counts(events, row, anchor)


def test_actual_sources_reject_mismatched_prefix_before_opening_any_frame():
    table = json.loads(m.TABLE.read_bytes())
    prepared = json.loads(m.H10_PREP.read_bytes())
    individual = json.loads(m.PHYS_PREP.read_bytes())
    receipt = json.loads(m.RECEIPT.read_bytes())
    individual["roster"][0]["prefix5_float32_hex"] = "00"*40
    with pytest.raises(ValueError, match="anchor identity"):
        m.join(table, prepared, individual, receipt, [None]*24, [None]*24, [None]*24)
