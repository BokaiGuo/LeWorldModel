import json

import pytest

from experiments.spectral_reachability import codex_p1_reset_envelope_overlap_20261030 as audit


def row(source, lo, hi, kind="candidate_random_reset_seed_envelope"):
    return dict(source_id=source, start_inclusive=lo, end_inclusive=hi, kind=kind, certified=False)


def test_inclusive_endpoints_and_reserved_are_not_discarded():
    rows = [row("A", 1, 3), row("B", 3, 5, "reserved_not_observed_reset_envelope"),
            row("C", 6, 9)]
    found = audit.overlaps(rows)
    assert len(found) == 1
    assert found[0]["overlap_inclusive"] == [3, 3]
    assert found[0]["integer_count"] == 1
    assert found[0]["right_kind"] == "reserved_not_observed_reset_envelope"


@pytest.mark.parametrize("bad", [
    row("A", 3, 2), row("A", -1, 2), row("A", True, 2),
    dict(row("A", 1, 2), certified=True), row("A", 1, 2, "invented"),
])
def test_invalid_envelopes_rejected(bad):
    with pytest.raises(ValueError):
        audit.overlaps([bad])


def test_duplicate_source_kind_rejected():
    with pytest.raises(ValueError):
        audit.overlaps([row("A", 0, 1), row("A", 2, 3)])


def test_real_index_and_uncertainty():
    value = audit.build()
    assert value["envelope_count"] == 11 and len(value["overlaps"]) == 11
    pair = next(p for p in value["overlaps"] if (p["left_source"], p["right_source"]) == ("E", "H"))
    assert pair["overlap_inclusive"] == [202753010, 202753041]
    assert value["prefix_pose_summary"]["records"] == 80
    assert value["prefix_pose_summary"]["distinct_exact_pose_keys"] == 8
    assert not value["reset_attempts_verified"] and not value["freshness_certified"]
    assert not value["P1_admission"] and value["training_disjoint"] == "unknown"


def test_hash_and_index_tampering_rejected(tmp_path, monkeypatch):
    original = audit.pose.INDEX
    data = json.loads(original.read_bytes())
    data["history_complete"] = True
    with pytest.raises(ValueError):
        audit.audit(data)
    path = tmp_path / "index.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(audit.pose, "INDEX", path)
    with pytest.raises(ValueError, match="digest"):
        audit.build()
