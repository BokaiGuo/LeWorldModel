import json

import pytest

from experiments.spectral_reachability import codex_p1_reset_sources_aggregate_20261101_exclusive as m


def test_saved_pairs():
    result = m.aggregate()
    assert not result["unknown_sources"]
    assert result["reset_records"] == 44
    assert sum(row.get("identity_status") == "original_attempt_not_independence_evidence" for row in result["events"]) == 16
    assert not result["complete_deduplication"] and not result["training_exclusion_certified"]
    assert all(row.get("completion") or row.get("completion_event") for row in result["events"])


def test_unknown_source_preserved(tmp_path):
    result = m.aggregate(tmp_path, {"missing": "0" * 64})
    assert result["reset_records"] == 0
    assert result["unknown_sources"][0]["status"] == "unknown"


def test_tampering_and_truncation_rejected():
    path = m.ROOT / "results/p1_observed_history_source_20261027_v1_main01/sourcepack.json"
    source, journal = path.read_bytes(), path.with_name("journal.jsonl").read_bytes()
    with pytest.raises(ValueError):
        m.extract_other(source, journal[:-1], str(path), "journal")
    with pytest.raises(ValueError):
        m.extract_other(source, journal.replace(b'"seed": 202781011', b'"seed": 202781012', 1), str(path), "journal")
    obj = json.loads(source)
    obj["counts"]["reset"]["attempted"] += 1
    with pytest.raises(ValueError):
        m.extract_other(json.dumps(obj).encode(), journal, str(path), "journal")


def test_exclusive_output(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "result.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["aggregate", "--output", str(output)])
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "keep"
